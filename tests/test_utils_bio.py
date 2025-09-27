from service.utils import (
    _extract_spans_from_bio,
    _spans_to_api_spans,
    _token_tags_to_char_bio,
)


def test_token_tags_to_char_bio_basic():
    text = "hello world"
    # token offsets for "hello" (0..5) and "world" (6..11)
    offsets = [(0, 5), (6, 11)]
    tags = ["B-TYPE", "O"]
    bio = _token_tags_to_char_bio(text, tags, offsets)
    assert bio[0] == "B-TYPE"
    assert all(x == "I-TYPE" for x in bio[1:5])
    assert bio[5] == "O"


def test_extract_spans_from_bio_and_api_spans():
    text = "hello world"
    # mark first word as TYPE
    bio = ["O"] * len(text)
    bio[0] = "B-TYPE"
    for i in range(1, 5):
        bio[i] = "I-TYPE"
    spans = _extract_spans_from_bio(text, bio)
    assert spans == [(0, 5, "TYPE")]
    api = _spans_to_api_spans(text, spans)
    assert api[0]["start_index"] == 0
    assert api[0]["end_index"] == 5
    assert api[0]["entity"].startswith("B-")
