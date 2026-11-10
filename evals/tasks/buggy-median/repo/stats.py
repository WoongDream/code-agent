"""간단한 통계 함수."""


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("빈 목록의 평균은 없다")
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("빈 목록의 중앙값은 없다")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid] + ordered[mid + 1]) / 2
