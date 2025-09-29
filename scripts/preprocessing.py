# from __future__ import annotations

# import re
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple

# # -----------------------------
# # Configuration for preprocessing
# # -----------------------------


# @dataclass
# class PreprocessConfig:
#     """Preprocessconfig."""

#     do_lower: bool = True
#     apply_translit_map: bool = True
#     max_seq_len: int = 128


# # -----------------------------
# # Preprocessor implementation
# # -----------------------------


# class Preprocessor:
#     """
#     Preprocessing utilities for the X5 NER pipeline.

#     Responsibilities:
#     - Text normalization (confusables, digit-letter fixes, safe substitutions)
#     - Conversions between spans and BIO at character level
#     - Alignment of char-level BIO to token-level labels using tokenizer offsets
#     - Decoding token-level predictions back to char-level BIO/spans and API spans
#     - Batch helpers for HF Datasets preparation
#     """

#     _lat = re.compile(r"[A-Za-z]")
#     _cyr = re.compile(r"[А-Яа-яЁё]")
#     _word_regex = re.compile(r"\S+")

#     INVISIBLE_SPACE_CHARS = {"\u00A0", "\u200B", "\u200C", "\u200D", "\ufeff"}

#     # Confusable mappings (Latin<->Cyrillic) — match notebook logic
#     LAT_TO_CYR = {
#         "A": "А",
#         "a": "а",
#         "B": "В",
#         "b": "в",
#         "C": "С",
#         "c": "с",
#         "E": "Е",
#         "e": "е",
#         "H": "Н",
#         "h": "н",
#         "K": "К",
#         "k": "к",
#         "M": "М",
#         "m": "м",
#         "O": "О",
#         "o": "о",
#         "P": "Р",
#         "p": "р",
#         "T": "Т",
#         "t": "т",
#         "X": "Х",
#         "x": "х",
#         "Y": "У",
#         "y": "у",
#     }
#     CYR_TO_LAT = {v: k for k, v in LAT_TO_CYR.items()}

#     # Digit->letter map — match notebook (smaller, conservative set)
#     DIGIT_TO_LETTER = {"0": "o", "1": "l", "3": "e", "5": "s"}

#     def __init__(
#         self,
#         tokenizer,
#         label2id: Dict[str, int],
#         id2label: Dict[int, str],
#         config: Optional[PreprocessConfig] = None,
#     ) -> None:
#         """Init.

#         Args:
#             self: Parameter.
#             tokenizer: Parameter.
#             label2id: Parameter.
#             id2label: Parameter.
#             config: Parameter.

#         Returns:
#             Return value.
#         """
#         self.tokenizer = tokenizer
#         self.label2id = label2id
#         self.id2label = id2label
#         self.config = config or PreprocessConfig()

#     # -----------------------------
#     # Normalization helpers
#     # -----------------------------

#     @classmethod
#     def _is_latin(cls, ch: str) -> bool:
#         """Is latin.

#         Args:
#             cls: Parameter.
#             ch: Parameter.

#         Returns:
#             Return value.
#         """
#         return bool(cls._lat.match(ch))

#     @classmethod
#     def _is_cyrillic(cls, ch: str) -> bool:
#         """Is cyrillic.

#         Args:
#             cls: Parameter.
#             ch: Parameter.

#         Returns:
#             Return value.
#         """
#         return bool(cls._cyr.match(ch))

#     @classmethod
#     def _token_script_stats(cls, tok: str) -> Tuple[int, int]:
#         """Token script stats.

#         Args:
#             cls: Parameter.
#             tok: Parameter.

#         Returns:
#             Return value.
#         """
#         lat = sum(1 for ch in tok if cls._is_latin(ch))
#         cyr = sum(1 for ch in tok if cls._is_cyrillic(ch))
#         return lat, cyr

#     @classmethod
#     def _normalize_mixed_script_token(cls, tok: str, apply_translit_map: bool) -> str:
#         """Normalize mixed script token.

#         Args:
#             cls: Parameter.
#             tok: Parameter.
#             apply_translit_map: Parameter.

#         Returns:
#             Return value.
#         """
#         lat, cyr = cls._token_script_stats(tok)
#         if lat == 0 or cyr == 0 or not apply_translit_map:
#             return tok
#         # If mixed, map minority script characters into the majority script
#         if cyr >= lat:
#             return "".join(cls.LAT_TO_CYR.get(ch, ch) for ch in tok)
#         return "".join(cls.CYR_TO_LAT.get(ch, ch) for ch in tok)

#     @classmethod
#     def _normalize_digit_letter_confusables(cls, tok: str) -> str:
#         """Normalize digit letter confusables.

#         Args:
#             cls: Parameter.
#             tok: Parameter.

#         Returns:
#             Return value.
#         """
#         if not tok:
#             return tok
#         out = []
#         for ch in tok:
#             if ch.isdigit():
#                 out.append(cls.DIGIT_TO_LETTER.get(ch, ch))
#             else:
#                 out.append(ch)
#         return "".join(out)

#     @classmethod
#     def _normalize_simple_subs(cls, tok: str) -> str:
#         """Normalize simple subs.

#         Args:
#             cls: Parameter.
#             tok: Parameter.

#         Returns:
#             Return value.
#         """
#         out = tok.replace("ё", "е").replace("Ё", "Е")
#         return out

#     def normalize(self, text: str) -> Tuple[str, List[int]]:
#         """
#         Normalize text and return (normalized_text, index_map_to_original).
#         Operations:
#         - Remove/replace invisible spaces
#         - Lowercasing (configurable)
#         - Simple substitutions (ё->е)
#         - Mixed-script cleanup and digit-letter confusables mapping per token
#         - Preserve a mapping from normalized indices to original indices
#         """
#         if text is None:
#             return "", []

#         # 1) Replace invisible spaces with ordinary space
#         stage_chars: List[str] = []
#         for ch in str(text):
#             if ch in self.INVISIBLE_SPACE_CHARS:
#                 stage_chars.append(" ")
#             else:
#                 stage_chars.append(ch)
#         stage = "".join(stage_chars)

#         # 2) Lowercase if configured
#         if self.config.do_lower:
#             stage = stage.lower()

#         # 3) Simple substitutions
#         stage = self._normalize_simple_subs(stage)

#         # 4) Token-level fixes (mixed scripts, digit-letter confusables)
#         out_chars: List[str] = []
#         index_map: List[int] = []
#         i = 0
#         while i < len(stage):
#             m = self._word_regex.match(stage, i)
#             if not m:
#                 # non-word char: copy as is
#                 out_chars.append(stage[i])
#                 index_map.append(i)
#                 i += 1
#                 continue
#             word = stage[m.start() : m.end()]
#             fixed = self._normalize_mixed_script_token(
#                 word, apply_translit_map=self.config.apply_translit_map
#             )
#             fixed = self._normalize_digit_letter_confusables(fixed)
#             for j, ch in enumerate(fixed):
#                 out_chars.append(ch)
#                 # map to first char of original segment + offset j (best-effort)
#                 index_map.append(m.start() + min(j, len(word) - 1))
#             i = m.end()

#         return "".join(out_chars), index_map

#     # -----------------------------
#     # BIO and spans helpers
#     # -----------------------------

#     @staticmethod
#     def to_char_bio(sample: str, spans: List[Tuple[int, int, str]]) -> List[str]:
#         """To char bio.

#         Args:
#             sample: Parameter.
#             spans: Parameter.

#         Returns:
#             Return value.
#         """
#         n = len(sample)
#         labels = ["O"] * n
#         for start, end, tag in spans:
#             if tag == "O" or "-" not in tag:
#                 continue
#             etype = tag.split("-", 1)[-1]
#             if start < 0 or end > n or start >= end:
#                 continue
#             labels[start] = f"B-{etype}"
#             for pos in range(start + 1, end):
#                 labels[pos] = f"I-{etype}"
#         return labels

#     def align_bio_to_tokens(self, text: str, char_bio: List[str]) -> Dict[str, Any]:
#         """Align bio to tokens.

#         Args:
#             self: Parameter.
#             text: Parameter.
#             char_bio: Parameter.

#         Returns:
#             Return value.
#         """
#         enc = self.tokenizer(
#             text,
#             return_offsets_mapping=True,
#             truncation=True,
#             max_length=self.config.max_seq_len,
#             padding=False,
#         )
#         offsets: List[Tuple[int, int]] = enc["offset_mapping"]
#         token_labels: List[int] = []

#         for start, end in offsets:
#             # Many tokenizers set (0,0) for special tokens
#             if end == 0 and start == 0:
#                 token_labels.append(-100)
#                 continue
#             if start == end:
#                 token_labels.append(-100)
#                 continue

#             # Determine label based on first labeled char within the span
#             tok_label = "O"
#             first_pos: Optional[int] = None
#             for pos in range(start, min(end, len(char_bio))):
#                 if char_bio[pos] != "O":
#                     tok_label = char_bio[pos]
#                     first_pos = pos
#                     break

#             if tok_label == "O":
#                 token_labels.append(self.label2id["O"])
#                 continue

#             bio, etype = tok_label.split("-", 1)
#             if (
#                 first_pos is not None
#                 and first_pos > 0
#                 and char_bio[first_pos - 1].endswith(etype)
#             ):
#                 tok_tag = f"I-{etype}"
#             else:
#                 tok_tag = f"B-{etype}"
#             token_labels.append(self.label2id.get(tok_tag, self.label2id["O"]))

#         enc["labels"] = token_labels
#         return enc

#     # -----------------------------
#     # Decoding helpers (token -> char BIO/spans)
#     # -----------------------------

#     @staticmethod
#     def token_tags_to_char_bio(
#         text: str, token_tags: List[str], offsets: List[Tuple[int, int]]
#     ) -> List[str]:
#         """Token tags to char bio.

#         Args:
#             text: Parameter.
#             token_tags: Parameter.
#             offsets: Parameter.

#         Returns:
#             Return value.
#         """
#         text_len = max([e for (_, e) in offsets if e is not None], default=0)
#         char_bio = ["O"] * text_len
#         for (start, end), tag in zip(offsets, token_tags):
#             if end is None or start is None or end <= start:
#                 continue
#             if tag == "O":
#                 continue
#             bio, etype = tag.split("-", 1)
#             # Head token -> B, others -> I
#             char_bio[start] = f"B-{etype}"
#             for pos in range(start + 1, min(end, text_len)):
#                 char_bio[pos] = f"I-{etype}"
#         return char_bio

#     @staticmethod
#     def decode_token_tags_to_char_spans(
#         text: str, token_tags: List[str], offsets: List[Tuple[int, int]]
#     ) -> List[Tuple[int, int, str]]:
#         """Decode token tags to char spans.

#         Args:
#             text: Parameter.
#             token_tags: Parameter.
#             offsets: Parameter.

#         Returns:
#             Return value.
#         """
#         text_len = max([e for (_, e) in offsets if e is not None], default=0)
#         char_bio = ["O"] * text_len
#         for (start, end), tag in zip(offsets, token_tags):
#             if end is None or start is None or end <= start:
#                 continue
#             if tag == "O":
#                 continue
#             bio, etype = tag.split("-", 1)
#             char_bio[start] = f"B-{etype}"
#             for pos in range(start + 1, min(end, text_len)):
#                 char_bio[pos] = f"I-{etype}"

#         # Extract spans from char BIO
#         spans: List[Tuple[int, int, str]] = []
#         i = 0
#         while i < len(char_bio):
#             if char_bio[i] == "O":
#                 i += 1
#                 continue
#             if char_bio[i].startswith("B-"):
#                 etype = char_bio[i].split("-", 1)[1]
#                 j = i + 1
#                 while j < len(char_bio) and char_bio[j] == f"I-{etype}":
#                     j += 1
#                 spans.append((i, j, etype))
#                 i = j
#                 continue
#             i += 1
#         return spans

#     # -----------------------------
#     # API spans and mapping helpers
#     # -----------------------------

#     @classmethod
#     def find_word_spans(cls, text: str) -> List[Tuple[int, int]]:
#         """Find word spans.

#         Args:
#             cls: Parameter.
#             text: Parameter.

#         Returns:
#             Return value.
#         """
#         return [(m.start(), m.end()) for m in cls._word_regex.finditer(text)]

#     @classmethod
#     def char_bio_to_api_spans(
#         cls, text: str, char_bio: List[str]
#     ) -> List[Dict[str, Any]]:
#         """Char bio to api spans.

#         Args:
#             cls: Parameter.
#             text: Parameter.
#             char_bio: Parameter.

#         Returns:
#             Return value.
#         """
#         n = len(char_bio)
#         if len(text) < n:
#             # pad text view if needed
#             text = text + " " * (n - len(text))
#         out: List[Dict[str, Any]] = []
#         for start, end in cls.find_word_spans(text[:n]):
#             # decide label for the word segment by majority/first non-O within range
#             tag = "O"
#             for pos in range(start, end):
#                 if pos < len(char_bio) and char_bio[pos] != "O":
#                     tag = char_bio[pos]
#                     break
#             if tag == "O":
#                 continue
#             out.append({"start_index": start, "end_index": end, "entity": tag})
#         return out

#     @staticmethod
#     def merge_adjacent_spans(
#         text: str, spans: List[Tuple[int, int, str]], max_gap: int = 1
#     ) -> List[Tuple[int, int, str]]:
#         """Merge adjacent spans.

#         Args:
#             text: Parameter.
#             spans: Parameter.
#             max_gap: Parameter.

#         Returns:
#             Return value.
#         """
#         spans_sorted = sorted(spans, key=lambda x: (x[2], x[0], x[1]))
#         out: List[Tuple[int, int, str]] = []
#         for s, e, t in spans_sorted:
#             if not out or out[-1][2] != t or s - out[-1][1] > max_gap:
#                 out.append((s, e, t))
#             else:
#                 ps, pe, pt = out[-1]
#                 out[-1] = (ps, max(pe, e), pt)
#         return out

#     @staticmethod
#     def map_spans_to_original(
#         spans: List[Tuple[int, int, str]], norm_to_orig: List[int]
#     ) -> List[Tuple[int, int, str]]:
#         """Map spans to original.

#         Args:
#             spans: Parameter.
#             norm_to_orig: Parameter.

#         Returns:
#             Return value.
#         """
#         mapped: List[Tuple[int, int, str]] = []
#         nmap = len(norm_to_orig)
#         for s, e, t in spans:
#             if s < 0 or e <= s:
#                 continue
#             if s >= nmap:
#                 continue
#             e_eff = min(e, nmap)
#             s_orig = norm_to_orig[s]
#             e_orig = norm_to_orig[e_eff - 1] + 1
#             mapped.append((s_orig, e_orig, t))
#         return mapped

#     @staticmethod
#     def spans_to_char_bio(
#         text_len: int, spans: List[Tuple[int, int, str]]
#     ) -> List[str]:
#         """Spans to char bio.

#         Args:
#             text_len: Parameter.
#             spans: Parameter.

#         Returns:
#             Return value.
#         """
#         bio = ["O"] * text_len
#         for s, e, t in spans:
#             if s < 0 or e <= s or e > text_len:
#                 continue
#             bio[s] = f"B-{t}"
#             for pos in range(s + 1, e):
#                 bio[pos] = f"I-{t}"
#         return bio

#     @classmethod
#     def spans_to_api_spans(
#         cls, text: str, spans: List[Tuple[int, int, str]], include_O: bool = False
#     ) -> List[Dict[str, Any]]:
#         """Spans to api spans.

#         Args:
#             cls: Parameter.
#             text: Parameter.
#             spans: Parameter.
#             include_O: Parameter.

#         Returns:
#             Return value.
#         """
#         merged = cls.merge_adjacent_spans(text, spans)
#         bio = cls.spans_to_char_bio(len(text), merged)
#         out: List[Dict[str, Any]] = []
#         for s, e in cls.find_word_spans(text):
#             tag = "O"
#             for pos in range(s, e):
#                 if bio[pos] != "O":
#                     tag = bio[pos]
#                     break
#             if tag == "O" and not include_O:
#                 continue
#             out.append({"start_index": s, "end_index": e, "entity": tag})
#         return out

#     # -----------------------------
#     # Batch helpers for HF Datasets
#     # -----------------------------

#     def build_char_bio_col(self, df) -> List[List[str]]:
#         """Build char bio col.

#         Args:
#             self: Parameter.
#             df: Parameter.

#         Returns:
#             Return value.
#         """
#         bios: List[List[str]] = []
#         for sample, ann in df[["sample", "parsed_annotation"]].itertuples(index=False):
#             bios.append(self.to_char_bio(sample, ann))
#         return bios

#     def encode_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
#         """Encode row.

#         Args:
#             self: Parameter.
#             row: Parameter.

#         Returns:
#             Return value.
#         """
#         enc = self.align_bio_to_tokens(row["sample"], row["char_bio"])
#         return {
#             "input_ids": enc["input_ids"],
#             "attention_mask": enc["attention_mask"],
#             "labels": enc["labels"],
#         }

#     def encode_dataframe(self, df) -> Any:
#         """Encode dataframe.

#         Args:
#             self: Parameter.
#             df: Parameter.

#         Returns:
#             Return value.
#         """
#         try:
#             from datasets import Dataset  # type: ignore
#         except Exception as e:
#             raise RuntimeError(
#                 "datasets library is required for encode_dataframe; install via pip install datasets"
#             ) from e

#         df = df.copy()
#         if "char_bio" not in df.columns:
#             df["char_bio"] = self.build_char_bio_col(df)
#         ds = Dataset.from_pandas(df[["sample", "char_bio"]])
#         ds = ds.map(lambda r: self.encode_row(r), remove_columns=ds.column_names)
#         ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#         return ds
