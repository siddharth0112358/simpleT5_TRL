import os
import inspect
import torch
import numpy as np
import pandas as pd
import warnings
from typing import Optional, List, Union, Literal, Callable, Any
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
# rouge_score is imported locally in _get_compute_metrics
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from trl import (
    DPOConfig,
    DPOTrainer,
    CPOConfig,
    CPOTrainer,
    SFTConfig,
    SFTTrainer,
)

torch.cuda.empty_cache()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class DataCollatorForSeq2SeqCompat(DataCollatorForSeq2Seq):
    """
    Custom DataCollatorForSeq2Seq that handles T5Gemma's different
    prepare_decoder_input_ids_from_labels signature.

    T5Gemma expects positional argument instead of keyword argument 'labels'.
    """

    def __call__(self, features, return_tensors=None):
        # For models that don't have prepare_decoder_input_ids_from_labels or have
        # a different signature (like T5Gemma), we handle decoder_input_ids manually
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            # Check if the model's method signature accepts 'labels' as keyword arg
            import inspect
            sig = inspect.signature(self.model.prepare_decoder_input_ids_from_labels)
            params = list(sig.parameters.keys())

            # If it only takes positional args (like T5Gemma), we need special handling
            if "labels" not in params or (len(params) == 1 and params[0] == "self"):
                # Don't pass model to parent, handle decoder_input_ids ourselves
                labels = [feature["labels"] for feature in features if "labels" in feature]

                # Call parent without model to avoid the problematic call
                original_model = self.model
                self.model = None
                batch = super().__call__(features, return_tensors=return_tensors)
                self.model = original_model

                # Now manually create decoder_input_ids if needed
                if labels and "decoder_input_ids" not in batch:
                    try:
                        # Try calling with positional argument
                        batch["decoder_input_ids"] = original_model.prepare_decoder_input_ids_from_labels(
                            batch["labels"]
                        )
                    except (TypeError, AttributeError):
                        # If that fails too, just shift the labels
                        decoder_start_token_id = getattr(original_model.config, "decoder_start_token_id", 0)
                        batch["decoder_input_ids"] = self._shift_right(batch["labels"], decoder_start_token_id)

                return batch

        # For standard models, use parent implementation
        return super().__call__(features, return_tensors=return_tensors)

    def _shift_right(self, labels, decoder_start_token_id):
        """Shift labels right to create decoder_input_ids."""
        import torch
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = decoder_start_token_id
        # Replace -100 (ignore index) with pad token
        shifted.masked_fill_(shifted == -100, self.tokenizer.pad_token_id if self.tokenizer else 0)
        return shifted


class Seq2SeqDPODataCollatorWithPadding:
    """
    Custom data collator for preference optimization training with Seq2Seq models.

    TRL's default DPODataCollatorWithPadding doesn't handle seq2seq-specific
    keys like 'answer_input_ids'. This collator extends support to handle
    these additional keys that are created during tokenization for encoder-decoder models.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = True,
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.is_encoder_decoder = is_encoder_decoder

    def __call__(self, features: list) -> dict:
        """Collate features into a batch with proper padding."""
        from torch.nn.utils.rnn import pad_sequence

        # Fill in missing attention masks, labels, and input ids derived from labels.
        for feature in features:
            # If tokenizers only provide labels, derive input_ids to keep TRL happy.
            label_to_input = {
                "chosen_labels": "chosen_input_ids",
                "rejected_labels": "rejected_input_ids",
                "completion_labels": "completion_input_ids",
            }
            for label_key, input_key in label_to_input.items():
                if input_key not in feature and label_key in feature and feature[label_key] is not None:
                    labels = feature[label_key]
                    if isinstance(labels, torch.Tensor):
                        input_ids = labels.clone()
                        input_ids[input_ids == self.label_pad_token_id] = self.pad_token_id
                    else:
                        if labels and isinstance(labels[0], (list, tuple)):
                            input_ids = [
                                [
                                    self.pad_token_id if t == self.label_pad_token_id else t
                                    for t in seq
                                ]
                                for seq in labels
                            ]
                        else:
                            input_ids = [
                                self.pad_token_id if t == self.label_pad_token_id else t
                                for t in labels
                            ]
                    feature[input_key] = input_ids

            # If labels are missing, derive them from input_ids.
            input_to_label = {
                "input_ids": "labels",
                "chosen_input_ids": "chosen_labels",
                "rejected_input_ids": "rejected_labels",
                "completion_input_ids": "completion_labels",
            }
            for input_key, label_key in input_to_label.items():
                if label_key in feature or input_key not in feature or feature[input_key] is None:
                    continue
                ids = feature[input_key]
                if isinstance(ids, torch.Tensor):
                    labels = ids.clone()
                    labels[labels == self.pad_token_id] = self.label_pad_token_id
                else:
                    if ids and isinstance(ids[0], (list, tuple)):
                        labels = [
                            [
                                self.label_pad_token_id if t == self.pad_token_id else t
                                for t in seq
                            ]
                            for seq in ids
                        ]
                    else:
                        labels = [
                            self.label_pad_token_id if t == self.pad_token_id else t
                            for t in ids
                        ]
                feature[label_key] = labels

            # Ensure attention masks exist for any input_ids fields.
            for key in list(feature.keys()):
                if key == "input_ids" or key.endswith("_input_ids"):
                    mask_key = "attention_mask" if key == "input_ids" else key.replace("_input_ids", "_attention_mask")
                    if mask_key in feature or feature.get(key) is None:
                        continue
                    ids = feature[key]
                    if not isinstance(ids, torch.Tensor):
                        ids = torch.tensor(ids)
                    feature[mask_key] = (ids != self.pad_token_id).long()

        # Explicit keys that need sequence padding with pad_token_id
        sequence_keys = {
            "input_ids", "attention_mask",
            "prompt_input_ids", "prompt_attention_mask",
            "completion_input_ids", "completion_attention_mask",
            "chosen_input_ids", "chosen_attention_mask",
            "rejected_input_ids", "rejected_attention_mask",
            "KL_completion_input_ids", "KL_completion_attention_mask",
            # Seq2seq specific keys
            "answer_input_ids", "answer_attention_mask",
            "decoder_input_ids", "decoder_attention_mask",
            "chosen_decoder_input_ids", "rejected_decoder_input_ids",
            "chosen_decoder_attention_mask", "rejected_decoder_attention_mask",
        }

        # Keys that need padding with label_pad_token_id
        label_keys = {"labels", "chosen_labels", "rejected_labels", "completion_labels"}

        # Scalar keys that should NOT be padded as sequences
        scalar_keys = {"ref_chosen_logps", "ref_rejected_logps", "label"}

        padded_batch = {}

        for k in features[0].keys():
            value = features[0][k]

            # Skip None values
            if value is None:
                continue
            # Skip raw text fields so they are not fed to the model
            if isinstance(value, str):
                continue

            # Handle scalar keys explicitly (logps, labels for KTO-style)
            if k in scalar_keys:
                padded_batch[k] = torch.tensor([f[k] for f in features])
                continue

            # Check if this is a sequence that needs padding
            is_sequence_key = k in sequence_keys or k in label_keys

            # For tensor values, check dimensionality
            if isinstance(value, torch.Tensor):
                is_sequence = value.dim() >= 1 and value.numel() > 1
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                is_sequence = not isinstance(value[0], (int, float, bool))
            else:
                is_sequence = False

            if is_sequence_key or (is_sequence and k not in scalar_keys):
                # Get all values for this key as tensors
                values = []
                for f in features:
                    v = f[k]
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v)
                    values.append(v)

                # Determine padding value
                if k in label_keys or k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                elif "attention_mask" in k:
                    padding_value = 0
                else:
                    padding_value = self.pad_token_id

                # Pad sequences
                padded_batch[k] = pad_sequence(values, batch_first=True, padding_value=padding_value)

            elif isinstance(value, (int, float, bool)):
                # Handle scalar values
                padded_batch[k] = torch.tensor([f[k] for f in features])
            elif isinstance(value, torch.Tensor) and value.dim() == 0:
                # Handle 0-dim tensors
                padded_batch[k] = torch.stack([f[k] for f in features])
            else:
                # Handle other types - try to convert to tensor
                try:
                    padded_batch[k] = torch.tensor([f[k] for f in features])
                except (TypeError, ValueError):
                    # If conversion fails, keep as list
                    padded_batch[k] = [f[k] for f in features]

        return padded_batch


class Seq2SeqTrainerCompat(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer that handles compatibility issues with older models.

    Some models (e.g., LongT5) don't accept the 'num_items_in_batch' argument
    that newer versions of transformers Trainer pass to compute_loss.
    This subclass filters out such arguments before passing to the model.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to filter out num_items_in_batch from inputs.

        The parent class may add num_items_in_batch to the inputs dict, but some
        models don't accept this parameter in their forward method.
        """
        # Remove num_items_in_batch from inputs if present, as some models don't accept it
        if "num_items_in_batch" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "num_items_in_batch"}

        # Some wrapped models (e.g., PEFT) accept **kwargs but the base model doesn't.
        # Avoid passing num_items_in_batch to the model forward.
        if hasattr(self, "model_accepts_loss_kwargs"):
            original_accepts = self.model_accepts_loss_kwargs
            self.model_accepts_loss_kwargs = False
            try:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                self.model_accepts_loss_kwargs = original_accepts

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )


def _is_prepared_preference_dataset(dataset: Dataset) -> bool:
    if dataset is None or not hasattr(dataset, "column_names"):
        return False
    required = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_labels",
        "rejected_input_ids",
        "rejected_attention_mask",
        "rejected_labels",
    }
    return required.issubset(set(dataset.column_names))


class Seq2SeqDPOTrainerCompat(DPOTrainer):
    """
    Skip TRL tokenization when the dataset is already tokenized for seq2seq DPO.
    """

    def _prepare_dataset(self, dataset, *args, **kwargs):
        if dataset is None:
            return None
        if _is_prepared_preference_dataset(dataset):
            return dataset
        return super()._prepare_dataset(dataset, *args, **kwargs)


class Seq2SeqCPOTrainerCompat(CPOTrainer):
    """
    Skip TRL tokenization when the dataset is already tokenized for seq2seq SimPO.
    """

    def _prepare_dataset(self, dataset, *args, **kwargs):
        if dataset is None:
            return None
        if _is_prepared_preference_dataset(dataset):
            return dataset
        return super()._prepare_dataset(dataset, *args, **kwargs)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_dataframe(df: pd.DataFrame, required_columns: list, method_name: str):
    """
    Validate DataFrame before passing to TRL trainers.
    Checks for missing columns, NaN values, and non-string types.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        method_name: Name of the calling method for error messages

    Raises:
        ValueError: If validation fails
    """
    # Check required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{method_name}: Missing required columns: {missing_cols}. "
            f"DataFrame must have columns: {required_columns}"
        )

    # Check for NaN values in required columns
    for col in required_columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            raise ValueError(
                f"{method_name}: Column '{col}' contains {nan_count} NaN values. "
                f"Please remove or fill NaN values before training."
            )

    # Check for non-string types in text columns
    for col in required_columns:
        non_string_mask = ~df[col].apply(lambda x: isinstance(x, str))
        non_string_count = non_string_mask.sum()
        if non_string_count > 0:
            raise ValueError(
                f"{method_name}: Column '{col}' contains {non_string_count} non-string values. "
                f"All text columns must contain strings. Use df['{col}'] = df['{col}'].astype(str) to convert."
            )


class SimpleT5_TRL:
    """Custom SimpleT5_TRL class for training encoder-decoder models using TRL/Transformers"""

    def __init__(self) -> None:
        """initiates SimpleT5_TRL class"""
        self.model = None
        self.tokenizer = None
        self.model_name = None  # Store model name for QLoRA reloading
        self.device = torch.device("cpu")
        self.finetuning = "full"
        self.trainer = None

    def _load_tokenizer(
        self,
        model_name: str,
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Load tokenizer for the model. Tries AutoTokenizer first, falls back to AutoProcessor
        for models like T5Gemma that require a processor.

        Args:
            model_name: Model name or path
            use_auth_token: HuggingFace auth token
            trust_remote_code: Allow custom code

        Returns:
            Tokenizer or Processor that can tokenize text
        """
        try:
            return AutoTokenizer.from_pretrained(
                model_name,
                token=use_auth_token,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError):
            # Some models like T5Gemma don't have a separate tokenizer config
            # and require AutoProcessor instead
            return AutoProcessor.from_pretrained(
                model_name,
                token=use_auth_token,
                trust_remote_code=trust_remote_code,
            )

    def from_pretrained(
        self,
        model_type_or_name: str = "t5-base",
        model_name: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        """
        Loads any encoder-decoder model for training/finetuning using AutoModelForSeq2SeqLM.
        Supports T5, MT5, ByT5, CodeT5, BART, T5Gemma, and any other Seq2Seq model.

        Args:
            model_type_or_name (str, optional): model name from HuggingFace Hub or local path.
                Examples: "t5-base", "google/mt5-base", "facebook/bart-base",
                "google/t5gemma-2-270m-270m", "google/t5gemma-2b-2b-ul2". Defaults to "t5-base".
                For backward compatibility, can also be model_type ("t5", "mt5", etc.)
                when model_name is provided as second argument.
            model_name (str, optional): For backward compatibility with v0.1.x API.
                If provided, model_type_or_name is treated as model_type and this is the actual model name.
            use_auth_token (str, optional): HuggingFace auth token for gated models. Defaults to None.
            trust_remote_code (bool, optional): Allow loading models with custom code. Defaults to False.
        """
        # Backward compatibility: if model_name is provided, use old API style
        if model_name is not None:
            actual_model_name = model_name
        else:
            actual_model_name = model_type_or_name

        # Store model name for later use (e.g., QLoRA reloading)
        self.model_name = actual_model_name

        self.tokenizer = self._load_tokenizer(
            actual_model_name,
            use_auth_token=use_auth_token,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            actual_model_name,
            token=use_auth_token,
            trust_remote_code=trust_remote_code,
        )
        self._ensure_num_hidden_layers()
        self._ensure_decoder_start_token_id()
        self._ensure_generation_config()

        # Warn about LongT5-tglobal numerical instability
        model_type = getattr(self.model.config, 'model_type', '').lower()
        if 'longt5' in model_type:
            encoder_attention_type = getattr(self.model.config, 'encoder_attention_type', '')
            if encoder_attention_type == 'transient-global' or 'tglobal' in actual_model_name.lower():
                warnings.warn(
                    "LongT5 with Transient Global Attention (tglobal) may produce NaN values "
                    "during training due to numerical instability. Consider using the local "
                    "variant instead (e.g., 'google/long-t5-local-base').",
                    UserWarning
                )

        # Ensure pad_token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Sync special token IDs between tokenizer and model config
        self._sync_special_tokens()

    def _ensure_generation_config(self, model: Optional[torch.nn.Module] = None):
        """
        Ensure model has proper generation config for seq2seq generation.

        Some models (e.g., LongT5) may not have all required generation settings,
        which can cause empty outputs during evaluation with predict_with_generate.
        """
        from transformers import GenerationConfig

        model = model or self.model
        if model is None:
            return

        # Get or create generation config
        if not hasattr(model, "generation_config") or model.generation_config is None:
            model.generation_config = GenerationConfig()

        gen_config = model.generation_config
        config = model.config

        # Ensure essential parameters are set
        if getattr(gen_config, "decoder_start_token_id", None) is None:
            gen_config.decoder_start_token_id = getattr(config, "decoder_start_token_id", None)

        if getattr(gen_config, "eos_token_id", None) is None:
            gen_config.eos_token_id = getattr(config, "eos_token_id", None)

        if getattr(gen_config, "pad_token_id", None) is None:
            gen_config.pad_token_id = getattr(config, "pad_token_id", None)
            if gen_config.pad_token_id is None and self.tokenizer is not None:
                gen_config.pad_token_id = self.tokenizer.pad_token_id

        # Ensure max_length has a reasonable default (but don't set max_new_tokens to avoid conflicts)
        if getattr(gen_config, "max_length", None) is None or gen_config.max_length == 20:
            gen_config.max_length = 512

    def _sync_special_tokens(self, model: Optional[torch.nn.Module] = None):
        """
        Sync special token IDs between tokenizer and model config.

        Ensures pad_token_id, eos_token_id, and bos_token_id are consistent
        between tokenizer and model, which is essential for proper loss computation.
        """
        model = model or self.model
        if model is None or self.tokenizer is None:
            return

        config = model.config

        # Sync pad_token_id
        if self.tokenizer.pad_token_id is not None:
            if getattr(config, "pad_token_id", None) != self.tokenizer.pad_token_id:
                config.pad_token_id = self.tokenizer.pad_token_id

        # Sync eos_token_id
        if self.tokenizer.eos_token_id is not None:
            if getattr(config, "eos_token_id", None) != self.tokenizer.eos_token_id:
                config.eos_token_id = self.tokenizer.eos_token_id

        # Sync bos_token_id if exists
        if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
            if getattr(config, "bos_token_id", None) != self.tokenizer.bos_token_id:
                config.bos_token_id = self.tokenizer.bos_token_id

    def _ensure_decoder_start_token_id(self, model: Optional[torch.nn.Module] = None):
        """
        Ensure config.decoder_start_token_id exists for encoder-decoder training.

        CPOTrainer/DPOTrainer require this for seq2seq models. If not set,
        defaults to pad_token_id or 0.
        """
        model = model or self.model
        if model is None or not hasattr(model, "config"):
            return
        config = model.config

        if getattr(config, "decoder_start_token_id", None) is not None:
            return

        # Try to get from tokenizer or use pad_token_id
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
                config.decoder_start_token_id = self.tokenizer.bos_token_id
            elif hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
                config.decoder_start_token_id = self.tokenizer.pad_token_id
            else:
                config.decoder_start_token_id = 0
        elif getattr(config, "pad_token_id", None) is not None:
            config.decoder_start_token_id = config.pad_token_id
        elif getattr(config, "bos_token_id", None) is not None:
            config.decoder_start_token_id = config.bos_token_id
        else:
            config.decoder_start_token_id = 0

    def _ensure_num_hidden_layers(self, model: Optional[torch.nn.Module] = None):
        """
        Ensure config.num_hidden_layers exists for generation cache compatibility.

        Some configs (e.g., T5Gemma/T5Gemma2) omit num_hidden_layers, which newer
        Transformers caching code expects.
        """
        model = model or self.model
        if model is None or not hasattr(model, "config"):
            return
        config = model.config
        if getattr(config, "num_hidden_layers", None) is not None:
            return

        key_priority = [
            "num_decoder_layers",
            "decoder_layers",
            "num_layers",
            "num_hidden_layers",
            "num_encoder_layers",
            "encoder_layers",
            "n_layer",
            "n_layers",
        ]

        def _normalize_value(value):
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, (list, tuple)) and value:
                return len(value)
            return None

        def _extract_from_dict(data):
            for key in key_priority:
                if key in data:
                    found = _normalize_value(data[key])
                    if found is not None:
                        return found
            for value in data.values():
                if isinstance(value, dict):
                    found = _extract_from_dict(value)
                    if found is not None:
                        return found
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            found = _extract_from_dict(item)
                            if found is not None:
                                return found
            return None

        def _extract_from_obj(obj):
            if obj is None:
                return None
            for key in key_priority:
                if hasattr(obj, key):
                    found = _normalize_value(getattr(obj, key, None))
                    if found is not None:
                        return found
            try:
                data = obj.to_dict()
            except Exception:
                data = None
            if isinstance(data, dict):
                return _extract_from_dict(data)
            return None

        nested_attrs = [
            "decoder_config",
            "text_config",
            "encoder_config",
            "model_config",
            "base_config",
            "base_model_config",
            "decoder",
            "encoder",
        ]

        value = None
        for attr in nested_attrs:
            value = _extract_from_obj(getattr(config, attr, None))
            if value is not None:
                break
        if value is None:
            value = _extract_from_obj(config)

        if value is not None:
            try:
                config.num_hidden_layers = value
            except Exception:
                if hasattr(config, "__dict__"):
                    config.__dict__["num_hidden_layers"] = value
        else:
            if hasattr(config, "use_cache"):
                config.use_cache = False
            if hasattr(model, "generation_config") and model.generation_config is not None:
                try:
                    model.generation_config.use_cache = False
                except Exception:
                    pass

    def _prepare_dataset(
        self,
        df: pd.DataFrame,
        source_max_token_len: int,
        target_max_token_len: int,
    ) -> Dataset:
        """Convert pandas DataFrame to HuggingFace Dataset with tokenization"""

        # Ensure pad_token_id is set
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            # Fallback: use eos_token_id or 0
            pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            warnings.warn(f"pad_token_id was None, using {pad_token_id} instead")

        def preprocess_function(examples):
            inputs = examples["source_text"]
            targets = examples["target_text"]

            model_inputs = self.tokenizer(
                inputs,
                max_length=source_max_token_len,
                padding="max_length",
                truncation=True,
            )

            # Use text_target for proper target tokenization in T5 models
            labels = self.tokenizer(
                text_target=targets,
                max_length=target_max_token_len,
                padding="max_length",
                truncation=True,
            )

            # Replace padding token id with -100 to ignore in loss calculation
            processed_labels = []
            for label in labels["input_ids"]:
                # Count non-padding tokens for validation
                non_pad_count = sum(1 for l in label if l != pad_token_id)
                if non_pad_count == 0:
                    warnings.warn("Found a label sequence with all padding tokens!")

                processed_label = [(l if l != pad_token_id else -100) for l in label]
                processed_labels.append(processed_label)

            model_inputs["labels"] = processed_labels
            return model_inputs

        # Convert DataFrame to Dataset
        dataset = Dataset.from_pandas(df[["source_text", "target_text"]])

        # Tokenize
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return tokenized_dataset

    def _get_compute_metrics(self):
        """
        Returns a compute_metrics function for evaluation with ROUGE scores.

        This is essential for T5/Seq2Seq models where loss alone is not informative.
        Returns ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores.
        """
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)

        def compute_metrics(eval_preds):
            predictions, labels = eval_preds

            # Decode predictions
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            # Ensure predictions is a numpy array
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)

            # Get pad_token_id, defaulting to 0 if not set
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0

            # Clip predictions to valid vocab range to prevent decode errors
            vocab_size = len(self.tokenizer)
            predictions = np.clip(predictions, 0, vocab_size - 1)

            # Replace -100 with pad_token_id for decoding
            predictions = np.where(predictions != -100, predictions, pad_token_id)
            labels = np.where(labels != -100, labels, pad_token_id)

            try:
                decoded_preds = self.tokenizer.batch_decode(
                    predictions, skip_special_tokens=True
                )
                decoded_labels = self.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
            except Exception as e:
                warnings.warn(f"Error decoding predictions: {e}. Returning zero scores.")
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

            # Strip whitespace
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]

            # Handle empty predictions/labels
            if not any(decoded_preds) or not any(decoded_labels):
                warnings.warn("Empty predictions or labels detected. Check generation config.")
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

            # Compute ROUGE scores using rouge_score directly
            try:
                rouge1_scores = []
                rouge2_scores = []
                rougeL_scores = []
                rougeLsum_scores = []

                for pred, label in zip(decoded_preds, decoded_labels):
                    if not pred or not label:
                        continue
                    scores = scorer.score(label, pred)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                    rougeLsum_scores.append(scores['rougeLsum'].fmeasure)

                if not rouge1_scores:
                    return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

                result = {
                    "rouge1": round(np.mean(rouge1_scores) * 100, 4),
                    "rouge2": round(np.mean(rouge2_scores) * 100, 4),
                    "rougeL": round(np.mean(rougeL_scores) * 100, 4),
                    "rougeLsum": round(np.mean(rougeLsum_scores) * 100, 4),
                }
            except Exception as e:
                warnings.warn(f"Error computing ROUGE: {e}. Returning zero scores.")
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

            return result

        return compute_metrics

    def _prepare_dpo_dataset(
        self,
        df: Optional[pd.DataFrame],
        max_prompt_length: int,
        max_completion_length: int,
    ) -> Optional[Dataset]:
        """
        Prepare DPO dataset with proper tokenization for encoder-decoder models.

        TRL's built-in tokenization may not create all required columns for encoder-decoder
        models (e.g., prompt_attention_mask). This method ensures proper tokenization.
        """
        if df is None:
            return None

        def tokenize_fn(examples):
            # Tokenize prompts (encoder inputs)
            prompt_tokens = self.tokenizer(
                examples["prompt"],
                max_length=max_prompt_length,
                padding="max_length",
                truncation=True,
            )

            # Tokenize chosen responses (decoder inputs/labels)
            chosen_tokens = self.tokenizer(
                examples["chosen"],
                max_length=max_completion_length,
                padding="max_length",
                truncation=True,
            )

            # Tokenize rejected responses (decoder inputs/labels)
            rejected_tokens = self.tokenizer(
                examples["rejected"],
                max_length=max_completion_length,
                padding="max_length",
                truncation=True,
            )

            # Create labels (replace pad tokens with -100)
            chosen_labels = [
                [(t if t != self.tokenizer.pad_token_id else -100) for t in tokens]
                for tokens in chosen_tokens["input_ids"]
            ]
            rejected_labels = [
                [(t if t != self.tokenizer.pad_token_id else -100) for t in tokens]
                for tokens in rejected_tokens["input_ids"]
            ]

            return {
                "prompt_input_ids": prompt_tokens["input_ids"],
                "prompt_attention_mask": prompt_tokens["attention_mask"],
                "chosen_input_ids": chosen_tokens["input_ids"],
                "chosen_attention_mask": chosen_tokens["attention_mask"],
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
                "rejected_labels": rejected_labels,
            }

        dataset = Dataset.from_pandas(df[["prompt", "chosen", "rejected"]])
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
        )
        return tokenized

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        # Data parameters
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 8,
        dataloader_num_workers: int = 2,
        # Training parameters
        max_epochs: int = 5,
        max_steps: int = -1,
        use_gpu: bool = True,
        precision: Union[str, int] = "32",
        seed: int = 42,
        # Checkpoint saving
        outputdir: str = "outputs",
        save_strategy: Literal["epoch", "steps", "no"] = "epoch",
        save_steps: int = 500,
        save_total_limit: Optional[int] = None,
        save_only_last_epoch: bool = False,
        # Evaluation
        eval_strategy: Literal["epoch", "steps", "no"] = "epoch",
        eval_steps: int = 500,
        # Logging
        logger: str = "tensorboard",
        logging_steps: int = 1,
        logging_dir: Optional[str] = None,
        # Early stopping
        early_stopping_patience_epochs: int = 0,
        load_best_model_at_end: bool = False,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        # Optimizer hyperparameters
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        optim: Literal["adamw_torch", "adamw_hf", "sgd", "adafactor"] = "adamw_torch",
        # Scheduler hyperparameters
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: Literal["linear", "cosine", "constant", "polynomial", "constant_with_warmup"] = "linear",
        # Gradient parameters
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        gradient_checkpointing: bool = False,
        # Finetuning type
        finetuning: Literal["full", "lora", "qlora"] = "full",
        # LoRA hyperparameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        lora_bias: Literal["none", "all", "lora_only"] = "none",
        # QLoRA quantization settings
        quantization: Literal["4bit", "8bit"] = "4bit",
        bit_loading: Optional[Literal["4bit", "8bit"]] = None,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        # Auth for gated models
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
        # Additional trainer arguments
        report_to: Optional[List[str]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Trains encoder-decoder model on custom dataset with support for full finetuning, LoRA, and QLoRA.

        Args:
            train_df (pd.DataFrame): training dataframe with columns "source_text" and "target_text"
            eval_df (pd.DataFrame, optional): validation dataframe with columns "source_text" and "target_text"

            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            dataloader_num_workers (int, optional): number of workers in dataloaders. Defaults to 2.

            max_epochs (int, optional): max number of epochs. Defaults to 5.
            max_steps (int, optional): max number of training steps. -1 means use max_epochs. Defaults to -1.
            use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
            precision (str/int, optional): training precision - "32", "16", "bf16". Defaults to "32".
            seed (int, optional): random seed for reproducibility. Defaults to 42.

            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            save_strategy (str, optional): when to save checkpoints - "epoch", "steps", "no". Defaults to "epoch".
            save_steps (int, optional): save checkpoint every N steps (when save_strategy="steps"). Defaults to 500.
            save_total_limit (int, optional): maximum number of checkpoints to keep. Defaults to None (keep all).
            save_only_last_epoch (bool, optional): if True, saves only the last epoch. Defaults to False.

            eval_strategy (str, optional): when to evaluate - "epoch", "steps", "no". Defaults to "epoch".
            eval_steps (int, optional): evaluate every N steps (when eval_strategy="steps"). Defaults to 500.

            logger (str, optional): logging backend - "tensorboard", "wandb", etc. Defaults to "tensorboard".
            logging_steps (int, optional): log every N steps. Defaults to 1.
            logging_dir (str, optional): directory for logs. Defaults to None (uses outputdir/runs).

            early_stopping_patience_epochs (int, optional): stops training if metric does not improve
                after specified evaluations. Set 0 to disable. Defaults to 0.
            load_best_model_at_end (bool, optional): load best model at end of training. Defaults to False.
            metric_for_best_model (str, optional): metric to monitor for best model. Defaults to "eval_loss".
            greater_is_better (bool, optional): whether higher metric is better. Defaults to False.

            learning_rate (float, optional): learning rate. Defaults to 1e-4.
            weight_decay (float, optional): weight decay for optimizer. Defaults to 0.0.
            adam_beta1 (float, optional): AdamW beta1. Defaults to 0.9.
            adam_beta2 (float, optional): AdamW beta2. Defaults to 0.999.
            adam_epsilon (float, optional): AdamW epsilon. Defaults to 1e-8.
            optim (str, optional): optimizer - "adamw_torch", "adamw_hf", "sgd", "adafactor". Defaults to "adamw_torch".

            warmup_steps (int, optional): number of warmup steps. Defaults to 0.
            warmup_ratio (float, optional): ratio of total steps for warmup (overrides warmup_steps). Defaults to 0.0.
            lr_scheduler_type (str, optional): scheduler - "linear", "cosine", "constant", "polynomial". Defaults to "linear".

            gradient_accumulation_steps (int, optional): accumulate gradients over n steps. Defaults to 1.
            gradient_clip_val (float, optional): gradient clipping value. Defaults to None.
            gradient_checkpointing (bool, optional): use gradient checkpointing to save memory. Defaults to False.

            finetuning (str, optional): finetuning type - "full", "lora", "qlora". Defaults to "full".

            lora_r (int, optional): LoRA rank. Defaults to 16.
            lora_alpha (int, optional): LoRA alpha parameter. Defaults to 32.
            lora_dropout (float, optional): LoRA dropout. Defaults to 0.05.
            lora_target_modules (list, optional): modules to apply LoRA. Defaults to None (auto-detect).
            lora_bias (str, optional): bias training - "none", "all", "lora_only". Defaults to "none".

            quantization (str, optional): quantization for QLoRA - "4bit" or "8bit". Defaults to "4bit".
            bit_loading (str, optional): alias for quantization ("4bit" or "8bit") to match QLoRA usage.
            bnb_4bit_compute_dtype (str, optional): compute dtype for 4bit. Defaults to "float16".
            bnb_4bit_quant_type (str, optional): quantization type - "nf4" or "fp4". Defaults to "nf4".
            bnb_4bit_use_double_quant (bool, optional): use double quantization. Defaults to True.

            use_auth_token (str, optional): HuggingFace auth token for gated models. Defaults to None.
            trust_remote_code (bool, optional): allow loading models with custom code. Defaults to False.

            report_to (list, optional): list of integrations to report to. Defaults to None.
            resume_from_checkpoint (str, optional): path to checkpoint to resume from. Defaults to None.
        """
        # Set seed
        set_seed(seed)

        # Validate input DataFrames
        _validate_dataframe(train_df, ["source_text", "target_text"], "train")
        if eval_df is not None:
            _validate_dataframe(eval_df, ["source_text", "target_text"], "train")

        self.finetuning = finetuning
        quantization_choice = bit_loading or quantization

        # Handle QLoRA - reload model with quantization
        if finetuning == "qlora":
            if self.model is None:
                raise ValueError("Please call from_pretrained() first before training.")

            # Get model name - prefer stored model_name, fallback to config
            model_name = self.model_name or getattr(self.model.config, '_name_or_path', None)
            if model_name is None:
                raise ValueError("Cannot determine model name for QLoRA reloading. Please call from_pretrained() first.")

            # Configure quantization
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.float16)

            if quantization_choice == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                )
            else:  # 8bit
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

            # Reload model with quantization
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=use_auth_token,
                trust_remote_code=trust_remote_code,
            )

            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            self._ensure_num_hidden_layers()
        self._ensure_decoder_start_token_id()

        # Enable gradient checkpointing
        if gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        # Apply LoRA for both lora and qlora
        if finetuning in ["lora", "qlora"]:
            # Auto-detect target modules if not specified
            if lora_target_modules is None:
                lora_target_modules = self._get_lora_target_modules()

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias=lora_bias,
                task_type=TaskType.SEQ_2_SEQ_LM,
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self._ensure_num_hidden_layers()
        self._ensure_decoder_start_token_id()
        self._ensure_generation_config()
        self._sync_special_tokens()

        # Prepare datasets
        train_dataset = self._prepare_dataset(train_df, source_max_token_len, target_max_token_len)
        eval_dataset = (
            self._prepare_dataset(eval_df, source_max_token_len, target_max_token_len)
            if eval_df is not None
            else None
        )

        # Data collator (use custom class for T5Gemma compatibility)
        data_collator = DataCollatorForSeq2SeqCompat(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        # Determine precision settings
        fp16 = False
        bf16 = False
        if precision in ["16", 16, "fp16", "16-mixed"]:
            fp16 = True
        elif precision in ["bf16", "bf16-mixed"]:
            bf16 = True

        # Handle save_only_last_epoch
        actual_save_strategy = save_strategy
        actual_save_steps = save_steps
        if save_only_last_epoch:
            actual_save_strategy = "epoch"

        # Logging directory (set via environment variable as logging_dir is deprecated)
        if logging_dir is None:
            logging_dir = os.path.join(outputdir, "runs")
        os.environ["TENSORBOARD_LOGGING_DIR"] = logging_dir

        # Report to
        if report_to is None:
            report_to = [logger] if logger != "default" else ["tensorboard"]

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=outputdir,

            # Training
            num_train_epochs=max_epochs if max_steps == -1 else 1,
            max_steps=max_steps if max_steps > 0 else -1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,

            # Optimizer
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            optim=optim,

            # Scheduler (warmup_ratio is deprecated, use warmup_steps only)
            warmup_steps=warmup_steps,
            lr_scheduler_type=lr_scheduler_type,

            # Gradient
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=gradient_clip_val if gradient_clip_val else 1.0,
            gradient_checkpointing=gradient_checkpointing,

            # Precision
            fp16=fp16,
            bf16=bf16,

            # Saving
            save_strategy=actual_save_strategy,
            save_steps=actual_save_steps,
            save_total_limit=save_total_limit,

            # Evaluation (disable if no eval dataset)
            eval_strategy=eval_strategy if eval_df is not None else "no",
            eval_steps=eval_steps if eval_strategy == "steps" and eval_df is not None else None,

            # Logging (logging_dir is deprecated, set via TENSORBOARD_LOGGING_DIR env var)
            logging_steps=logging_steps,
            report_to=report_to,

            # Best model
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,

            # Other
            seed=seed,
            dataloader_num_workers=dataloader_num_workers,
            remove_unused_columns=False,
            use_cpu=not use_gpu,

            # Generation config for eval (required for ROUGE evaluation)
            predict_with_generate=True,
            generation_max_length=target_max_token_len,
        )

        # Early stopping callback
        callbacks = []
        if early_stopping_patience_epochs > 0:
            from transformers import EarlyStoppingCallback
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_epochs))

        # Create trainer with compute_metrics for ROUGE evaluation
        # Use Seq2SeqTrainerCompat for compatibility with models that don't accept num_items_in_batch
        self.trainer = Seq2SeqTrainerCompat(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._get_compute_metrics() if eval_dataset is not None else None,
            callbacks=callbacks if callbacks else None,
        )

        # Train
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        if save_only_last_epoch:
            final_path = os.path.join(outputdir, "final_model")
            self._save_model(final_path)

    def _get_lora_target_modules(self) -> List[str]:
        """
        Auto-detect appropriate LoRA target modules based on model architecture.

        For encoder-decoder models, this targets both self-attention and cross-attention
        layers. In T5-style models, cross-attention (EncDecAttention) uses the same
        naming convention as self-attention, so targeting q, k, v, o applies LoRA
        to both self-attention and cross-attention layers for optimal performance.
        """
        model_type = getattr(self.model.config, 'model_type', '').lower()

        # Target modules for encoder-decoder models
        # Include q, k, v, o to cover both self-attention and cross-attention layers
        target_modules_map = {
            't5gemma': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],  # T5Gemma uses Gemma naming
            't5gemma2': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            't5': ['q', 'k', 'v', 'o'],  # Covers SelfAttention + EncDecAttention
            'mt5': ['q', 'k', 'v', 'o'],
            'bart': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'mbart': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'pegasus': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'led': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'blenderbot': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'marian': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        }

        # Try to match model type
        for key, modules in target_modules_map.items():
            if key in model_type:
                return modules

        # Default fallback - try common patterns for both self and cross attention
        return ['q', 'k', 'v', 'o', 'q_proj', 'k_proj', 'v_proj', 'out_proj']

    def _save_model(self, path: str):
        """Save model and tokenizer to path"""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    def load_model(
        self,
        model_type_or_dir: str,
        model_dir: Optional[str] = None,
        use_gpu: bool = False,
        finetuning: Literal["full", "lora", "qlora"] = "full",
        base_model_name: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
        # Quantization for QLoRA inference
        quantization: Optional[Literal["4bit", "8bit"]] = None,
        bit_loading: Optional[Literal["4bit", "8bit"]] = None,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ):
        """
        Loads a checkpoint for inferencing/prediction

        Args:
            model_type_or_dir (str): path to model directory.
                For backward compatibility, can also be model_type ("t5", "mt5", etc.)
                when model_dir is provided as second argument.
            model_dir (str, optional): For backward compatibility with v0.1.x API.
                If provided, model_type_or_dir is ignored and this is used as the model directory.
            use_gpu (bool, optional): if True, model uses gpu for inference. Defaults to False.
            finetuning (str, optional): finetuning type used for training - "full", "lora", "qlora".
                Defaults to "full".
            base_model_name (str, optional): base model name for LoRA/QLoRA. Required for lora/qlora.
            use_auth_token (str, optional): HuggingFace auth token for gated models. Defaults to None.
            trust_remote_code (bool, optional): allow loading models with custom code. Defaults to False.
            quantization (str, optional): quantization for inference - "4bit", "8bit", or None for no
                quantization. Defaults to None.
            bit_loading (str, optional): alias for quantization ("4bit" or "8bit") when loading QLoRA models.
            bnb_4bit_compute_dtype (str, optional): compute dtype for 4bit. Defaults to "float16".
            bnb_4bit_quant_type (str, optional): quantization type - "nf4" or "fp4". Defaults to "nf4".
            bnb_4bit_use_double_quant (bool, optional): use double quantization. Defaults to True.
        """
        # Backward compatibility: if model_dir is provided, use old API style
        if model_dir is not None:
            actual_model_dir = model_dir
        else:
            actual_model_dir = model_type_or_dir
        self.finetuning = finetuning
        quantization_choice = bit_loading or quantization

        if finetuning in ["lora", "qlora"]:
            if base_model_name is None:
                raise ValueError("base_model_name is required for loading LoRA/QLoRA models")

            # Load base model (with optional quantization)
            if quantization_choice:
                compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.float16)

                if quantization_choice == "4bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_quant_type=bnb_4bit_quant_type,
                        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    )
                else:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=use_auth_token,
                    trust_remote_code=trust_remote_code,
                    use_safetensors=False,
                )
            else:
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    base_model_name,
                    token=use_auth_token,
                    trust_remote_code=trust_remote_code,
                    use_safetensors=False,
                )
            self._ensure_num_hidden_layers(base_model)
            self._ensure_decoder_start_token_id(base_model)

            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, actual_model_dir)
            self._ensure_num_hidden_layers(self.model)
            self._ensure_decoder_start_token_id(self.model)
            self._ensure_generation_config(self.model)
            self.tokenizer = self._load_tokenizer(
                actual_model_dir, trust_remote_code=trust_remote_code
            )
            # Note: _sync_special_tokens is called after the common pad_token check below
        else:
            # Load full model
            if quantization_choice:
                compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.float16)

                if quantization_choice == "4bit":
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_quant_type=bnb_4bit_quant_type,
                        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                    )
                else:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    actual_model_dir,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=trust_remote_code,
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    actual_model_dir,
                    trust_remote_code=trust_remote_code,
                )
            self._ensure_num_hidden_layers()
            self._ensure_decoder_start_token_id()
            self._ensure_generation_config()
            self.tokenizer = self._load_tokenizer(
                actual_model_dir, trust_remote_code=trust_remote_code
            )

        # Ensure pad_token exists for tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Sync special token IDs between tokenizer and model config
        self._sync_special_tokens()

        # Set device
        if quantization_choice:
            # For quantized models, device_map handles device placement
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                raise RuntimeError("No GPU found. Set use_gpu=False to use CPU")
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)

    def merge_lora_weights(self):
        """
        Merge LoRA weights into the base model for faster inference.
        Only applicable for LoRA/QLoRA models.
        """
        if self.finetuning not in ["lora", "qlora"]:
            print("Model is not a LoRA/QLoRA model. No merging needed.")
            return

        if not isinstance(self.model, PeftModel):
            print("Model is not a PEFT model. No merging needed.")
            return

        self.model = self.model.merge_and_unload()
        print("LoRA weights merged successfully.")

    def save_merged_model(self, output_dir: str):
        """
        Save merged model (LoRA weights merged into base model).
        Useful for deployment without PEFT dependency.

        Args:
            output_dir (str): directory to save the merged model
        """
        if isinstance(self.model, PeftModel):
            self.merge_lora_weights()

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Merged model saved to {output_dir}")

    # ==================== DPO Training ====================

    def train_dpo(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        # Data parameters
        max_length: int = 512,
        max_prompt_length: int = 256,
        max_completion_length: int = 256,
        batch_size: int = 4,
        # Training parameters
        max_epochs: int = 3,
        max_steps: int = -1,
        use_gpu: bool = True,
        precision: Union[str, int] = "32",
        seed: int = 42,
        # DPO specific
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair", "bco_pair", "robust"] = "sigmoid",
        label_smoothing: float = 0.0,
        # Checkpoint saving
        outputdir: str = "outputs",
        save_strategy: Literal["epoch", "steps", "no"] = "epoch",
        save_steps: int = 500,
        save_total_limit: Optional[int] = None,
        # Evaluation
        eval_strategy: Literal["epoch", "steps", "no"] = "epoch",
        eval_steps: int = 500,
        # Logging
        logging_steps: int = 1,
        report_to: Optional[List[str]] = None,
        # Optimizer
        learning_rate: float = 5e-7,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: str = "linear",
        # Gradient
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        # Finetuning type
        finetuning: Literal["full", "lora", "qlora"] = "full",
        # LoRA hyperparameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # QLoRA quantization settings
        quantization: Literal["4bit", "8bit"] = "4bit",
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        # Auth
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
        # Resume
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Train model using Direct Preference Optimization (DPO).

        DPO is a simpler alternative to RLHF that directly optimizes for human preferences
        without needing a separate reward model.

        Args:
            train_df (pd.DataFrame): DataFrame with columns "prompt", "chosen", "rejected"
            eval_df (pd.DataFrame, optional): Evaluation DataFrame with same columns
            max_length (int): Maximum sequence length. Defaults to 512.
            max_prompt_length (int): Maximum prompt length. Defaults to 256.
            batch_size (int): Batch size. Defaults to 4.
            max_epochs (int): Number of training epochs. Defaults to 3.
            beta (float): DPO beta parameter controlling deviation from reference. Defaults to 0.1.
            loss_type (str): DPO loss variant. Defaults to "sigmoid".
            label_smoothing (float): Label smoothing for DPO. Defaults to 0.0.
            finetuning (str): "full", "lora", or "qlora". Defaults to "full".
            ... (other parameters same as train method)

        Note:
            The use_gpu parameter is accepted for API consistency but device placement
            is handled automatically by the TRL Trainer.
        """
        set_seed(seed)
        self.finetuning = finetuning

        # Validate input DataFrames
        _validate_dataframe(train_df, ["prompt", "chosen", "rejected"], "train_dpo")
        if eval_df is not None:
            _validate_dataframe(eval_df, ["prompt", "chosen", "rejected"], "train_dpo")

        # Handle QLoRA
        if finetuning == "qlora":
            model_name = self.model_name or getattr(self.model.config, '_name_or_path', None)
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.float16)

            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=use_auth_token,
                trust_remote_code=trust_remote_code,
            )
            self.model = prepare_model_for_kbit_training(self.model)
            self._ensure_num_hidden_layers()
        self._ensure_decoder_start_token_id()

        if gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        # Apply LoRA
        if finetuning in ["lora", "qlora"]:
            if lora_target_modules is None:
                lora_target_modules = self._get_lora_target_modules()

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Prepare datasets - pass raw text, let DPOTrainer tokenize
        # The custom data collator will synthesize any missing attention masks
        train_dataset = Dataset.from_pandas(train_df[["prompt", "chosen", "rejected"]])
        eval_dataset = Dataset.from_pandas(eval_df[["prompt", "chosen", "rejected"]]) if eval_df is not None else None

        # Precision
        fp16 = precision in ["16", 16, "fp16", "16-mixed"]
        bf16 = precision in ["bf16", "bf16-mixed"]

        if report_to is None:
            report_to = ["tensorboard"]

        # Disable eval_strategy if no eval_df provided
        if eval_df is None and eval_strategy != "no":
            eval_strategy = "no"

        # DPO Config
        dpo_config = DPOConfig(
            output_dir=outputdir,
            num_train_epochs=max_epochs if max_steps == -1 else 1,
            max_steps=max_steps if max_steps > 0 else -1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
                        lr_scheduler_type=lr_scheduler_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            use_cpu=not use_gpu,
            fp16=fp16,
            bf16=bf16,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps if eval_strategy == "steps" else None,
            logging_steps=logging_steps,
            report_to=report_to,
            seed=seed,
            beta=beta,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            remove_unused_columns=False,  # Required for encoder-decoder models
        )

        # Ensure num_hidden_layers exists for T5Gemma compatibility
        self._ensure_num_hidden_layers()
        self._ensure_decoder_start_token_id()

        # Create custom data collator for seq2seq models
        data_collator = Seq2SeqDPODataCollatorWithPadding(
            pad_token_id=self.tokenizer.pad_token_id,
            label_pad_token_id=-100,
            is_encoder_decoder=True,
        )

        # Create trainer
        self.trainer = Seq2SeqDPOTrainerCompat(
            model=self.model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        if hasattr(self.trainer, "args") and hasattr(self.trainer.args, "remove_unused_columns"):
            self.trainer.args.remove_unused_columns = False

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ==================== SimPO Training ====================

    def train_simpo(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        # Data parameters
        max_length: int = 512,
        max_prompt_length: int = 256,
        max_completion_length: int = 256,
        batch_size: int = 4,
        # Training parameters
        max_epochs: int = 3,
        max_steps: int = -1,
        use_gpu: bool = True,
        precision: Union[str, int] = "32",
        seed: int = 42,
        # SimPO specific
        beta: float = 2.0,
        simpo_gamma: float = 0.5,
        label_smoothing: float = 0.0,
        # Checkpoint saving
        outputdir: str = "outputs",
        save_strategy: Literal["epoch", "steps", "no"] = "epoch",
        save_steps: int = 500,
        save_total_limit: Optional[int] = None,
        # Evaluation
        eval_strategy: Literal["epoch", "steps", "no"] = "epoch",
        eval_steps: int = 500,
        # Logging
        logging_steps: int = 1,
        report_to: Optional[List[str]] = None,
        # Optimizer
        learning_rate: float = 5e-7,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: str = "linear",
        # Gradient
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        # Finetuning type
        finetuning: Literal["full", "lora", "qlora"] = "full",
        # LoRA hyperparameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # QLoRA quantization settings
        quantization: Literal["4bit", "8bit"] = "4bit",
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        # Auth
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
        # Resume
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Train model using Simple Preference Optimization (SimPO).

        SimPO is a simpler variant of DPO that uses length-normalized rewards
        and doesn't require a reference model, making it more memory efficient.

        Args:
            train_df (pd.DataFrame): DataFrame with columns "prompt", "chosen", "rejected"
            eval_df (pd.DataFrame, optional): Evaluation DataFrame with same columns
            beta (float): SimPO beta parameter. Defaults to 2.0.
            simpo_gamma (float): SimPO gamma (target reward margin). Defaults to 0.5.
            label_smoothing (float): Label smoothing. Defaults to 0.0.
            ... (other parameters same as train method)

        Note:
            The use_gpu parameter is accepted for API consistency but device placement
            is handled automatically by the TRL Trainer. SimPO is reference-free by design.
        """
        set_seed(seed)
        self.finetuning = finetuning

        # Validate input DataFrames
        _validate_dataframe(train_df, ["prompt", "chosen", "rejected"], "train_simpo")
        if eval_df is not None:
            _validate_dataframe(eval_df, ["prompt", "chosen", "rejected"], "train_simpo")

        # Handle QLoRA
        if finetuning == "qlora":
            model_name = self.model_name or getattr(self.model.config, '_name_or_path', None)
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.float16)

            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=use_auth_token,
                trust_remote_code=trust_remote_code,
            )
            self.model = prepare_model_for_kbit_training(self.model)

        if gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        # Apply LoRA
        if finetuning in ["lora", "qlora"]:
            if lora_target_modules is None:
                lora_target_modules = self._get_lora_target_modules()

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Prepare datasets - pass raw text, let CPOTrainer tokenize
        # The custom data collator will synthesize any missing attention masks
        train_dataset = Dataset.from_pandas(train_df[["prompt", "chosen", "rejected"]])
        eval_dataset = Dataset.from_pandas(eval_df[["prompt", "chosen", "rejected"]]) if eval_df is not None else None

        # Precision
        fp16 = precision in ["16", 16, "fp16", "16-mixed"]
        bf16 = precision in ["bf16", "bf16-mixed"]

        if report_to is None:
            report_to = ["tensorboard"]

        # Disable eval_strategy if no eval_df provided
        if eval_df is None and eval_strategy != "no":
            eval_strategy = "no"

        # CPO Config with SimPO loss
        cpo_config = CPOConfig(
            output_dir=outputdir,
            num_train_epochs=max_epochs if max_steps == -1 else 1,
            max_steps=max_steps if max_steps > 0 else -1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
                        lr_scheduler_type=lr_scheduler_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            use_cpu=not use_gpu,
            fp16=fp16,
            bf16=bf16,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps if eval_strategy == "steps" else None,
            logging_steps=logging_steps,
            report_to=report_to,
            seed=seed,
            beta=beta,
            loss_type="simpo",
            simpo_gamma=simpo_gamma,
            label_smoothing=label_smoothing,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            remove_unused_columns=False,  # Required for encoder-decoder models
        )

        # Ensure num_hidden_layers exists for T5Gemma compatibility
        self._ensure_num_hidden_layers()
        self._ensure_decoder_start_token_id()

        # Create custom data collator for seq2seq models
        data_collator = Seq2SeqDPODataCollatorWithPadding(
            pad_token_id=self.tokenizer.pad_token_id,
            label_pad_token_id=-100,
            is_encoder_decoder=True,
        )

        # Create trainer using CPOTrainer with SimPO loss
        self.trainer = Seq2SeqCPOTrainerCompat(
            model=self.model,
            args=cpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        if hasattr(self.trainer, "args") and hasattr(self.trainer.args, "remove_unused_columns"):
            self.trainer.args.remove_unused_columns = False

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ==================== RFT Training ====================

    def train_rft(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        # Data parameters
        max_seq_length: int = 512,
        batch_size: int = 8,
        # Training parameters
        max_epochs: int = 3,
        max_steps: int = -1,
        use_gpu: bool = True,
        precision: Union[str, int] = "32",
        seed: int = 42,
        # Checkpoint saving
        outputdir: str = "outputs",
        save_strategy: Literal["epoch", "steps", "no"] = "epoch",
        save_steps: int = 500,
        save_total_limit: Optional[int] = None,
        # Evaluation
        eval_strategy: Literal["epoch", "steps", "no"] = "epoch",
        eval_steps: int = 500,
        # Logging
        logging_steps: int = 1,
        report_to: Optional[List[str]] = None,
        # Optimizer
        learning_rate: float = 2e-5,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        warmup_ratio: float = 0.0,
        lr_scheduler_type: str = "linear",
        # Gradient
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        # Finetuning type
        finetuning: Literal["full", "lora", "qlora"] = "full",
        # LoRA hyperparameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        # QLoRA quantization settings
        quantization: Literal["4bit", "8bit"] = "4bit",
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        # RFT specific
        packing: bool = False,
        dataset_text_field: str = "text",
        # Auth
        use_auth_token: Optional[str] = None,
        trust_remote_code: bool = False,
        # Resume
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Train model using Reinforcement Fine-Tuning (RFT).

        RFT is supervised fine-tuning on high-quality samples, typically filtered
        by a reward model. This is the simplest form of aligning models with
        human preferences - just train on the best examples.

        For encoder-decoder models, use standard source_text/target_text format.
        For text completion style, provide "text" column with full examples.

        Args:
            train_df (pd.DataFrame): DataFrame with either:
                - "source_text" and "target_text" columns (encoder-decoder style)
                - "text" column with complete examples (completion style)
            eval_df (pd.DataFrame, optional): Evaluation DataFrame with same columns
            max_seq_length (int): Maximum sequence length. Defaults to 512.
            packing (bool): Pack multiple short examples into one sequence. Defaults to False.
            dataset_text_field (str): Column name for text data. Defaults to "text".
            ... (other parameters same as train method)

        Note:
            The use_gpu parameter is accepted for API consistency but device placement
            is handled automatically by the TRL Trainer.
        """
        set_seed(seed)
        self.finetuning = finetuning

        # Check data format - if source_text/target_text exist, use seq2seq format
        use_seq2seq = "source_text" in train_df.columns and "target_text" in train_df.columns

        # Validate input DataFrames
        if use_seq2seq:
            _validate_dataframe(train_df, ["source_text", "target_text"], "train_rft")
            if eval_df is not None:
                _validate_dataframe(eval_df, ["source_text", "target_text"], "train_rft")
        else:
            _validate_dataframe(train_df, [dataset_text_field], "train_rft")
            if eval_df is not None:
                _validate_dataframe(eval_df, [dataset_text_field], "train_rft")

        if use_seq2seq:
            # For seq2seq models, use the standard train method
            return self.train(
                train_df=train_df,
                eval_df=eval_df,
                source_max_token_len=max_seq_length,
                target_max_token_len=max_seq_length,
                batch_size=batch_size,
                max_epochs=max_epochs,
                max_steps=max_steps,
                use_gpu=use_gpu,
                precision=precision,
                seed=seed,
                outputdir=outputdir,
                save_strategy=save_strategy,
                save_steps=save_steps,
                save_total_limit=save_total_limit,
                eval_strategy=eval_strategy,
                eval_steps=eval_steps,
                logging_steps=logging_steps,
                report_to=report_to,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                                lr_scheduler_type=lr_scheduler_type,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                finetuning=finetuning,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                quantization=quantization,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                use_auth_token=use_auth_token,
                trust_remote_code=trust_remote_code,
                resume_from_checkpoint=resume_from_checkpoint,
            )

        # For text completion style, use SFTTrainer
        # Handle QLoRA
        if finetuning == "qlora":
            model_name = self.model_name or getattr(self.model.config, '_name_or_path', None)
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.float16)

            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=use_auth_token,
                trust_remote_code=trust_remote_code,
            )
            self.model = prepare_model_for_kbit_training(self.model)

        if gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        # Apply LoRA
        if finetuning in ["lora", "qlora"]:
            if lora_target_modules is None:
                lora_target_modules = self._get_lora_target_modules()

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Ensure num_hidden_layers exists for T5Gemma compatibility
        self._ensure_num_hidden_layers()
        self._ensure_decoder_start_token_id()

        # Prepare dataset
        train_dataset = Dataset.from_pandas(train_df[[dataset_text_field]])
        eval_dataset = Dataset.from_pandas(eval_df[[dataset_text_field]]) if eval_df is not None else None

        # Precision
        fp16 = precision in ["16", 16, "fp16", "16-mixed"]
        bf16 = precision in ["bf16", "bf16-mixed"]

        if report_to is None:
            report_to = ["tensorboard"]

        # Disable eval_strategy if no eval_df provided
        if eval_df is None and eval_strategy != "no":
            eval_strategy = "no"

        # SFT Config
        sft_config = SFTConfig(
            output_dir=outputdir,
            num_train_epochs=max_epochs if max_steps == -1 else 1,
            max_steps=max_steps if max_steps > 0 else -1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
                        lr_scheduler_type=lr_scheduler_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            use_cpu=not use_gpu,
            fp16=fp16,
            bf16=bf16,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps if eval_strategy == "steps" else None,
            logging_steps=logging_steps,
            report_to=report_to,
            seed=seed,
            max_seq_length=max_seq_length,
            packing=packing,
            dataset_text_field=dataset_text_field,
        )

        # Create trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        min_length: int = 0,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        temperature: float = 1.0,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        encoder_no_repeat_ngram_size: int = 0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        use_cache: bool = True,
    ) -> List[str]:
        """
        Generates prediction for encoder-decoder model.

        Args:
            source_text (str): input text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            min_length (int, optional): min token length of prediction. Defaults to 0.
            num_return_sequences (int, optional): number of predictions to return. Defaults to 1.
            num_beams (int, optional): number of beams for beam search. Defaults to 2.
            top_k (int, optional): top-k sampling parameter. Defaults to 50.
            top_p (float, optional): top-p (nucleus) sampling parameter. Defaults to 0.95.
            do_sample (bool, optional): whether to use sampling. Defaults to True.
            temperature (float, optional): sampling temperature. Defaults to 1.0.
            repetition_penalty (float, optional): penalty for repetition. Defaults to 2.5.
            length_penalty (float, optional): length penalty for beam search. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): prevent repeating n-grams of this size. Defaults to 0.
            encoder_no_repeat_ngram_size (int, optional): prevent copying n-grams from input. Defaults to 0.
            early_stopping (bool, optional): stop beam search early. Defaults to True.
            skip_special_tokens (bool, optional): skip special tokens in output. Defaults to True.
            clean_up_tokenization_spaces (bool, optional): clean up spaces. Defaults to True.
            use_cache (bool, optional): use KV cache for faster generation. Defaults to True.

        Returns:
            list[str]: list of generated predictions
        """
        self.model.eval()

        # Ensure generation config is set up properly
        self._ensure_generation_config()

        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self.device)

        # Get pad_token_id for generation
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        # Get decoder_start_token_id
        decoder_start_token_id = getattr(self.model.config, "decoder_start_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                temperature=temperature,
                no_repeat_ngram_size=no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                num_return_sequences=num_return_sequences,
                use_cache=use_cache,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
            )

        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]
        return preds
