from wordcount import count_words, main


def test_count_words_lowercases():
    assert count_words("Apple apple BANANA") == {"apple": 2, "banana": 1}


def test_main_prints_most_common_first(tmp_path, capsys):
    path = tmp_path / "words.txt"
    path.write_text("b a b c b a", encoding="utf-8")
    assert main([str(path)]) == 0
    assert capsys.readouterr().out.splitlines() == ["b 3", "a 2", "c 1"]
