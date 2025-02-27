from shapely import Point, Polygon

def halton_sequence(b):
    """Generator function for Halton sequence."""
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d

def halton_2d(b1=2, b2=3):
    """Generate 2D Halton sequence."""
    seq1, seq2 = halton_sequence(b1), halton_sequence(b2)
    while True:
        yield (next(seq1), next(seq2))

def geo_halton_2d(polygon: Polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    delta_x = max_x - min_x
    delta_y = max_y - min_y

    seq = halton_2d()
    while True:
        x, y = next(seq)
        x = min_x + x * delta_x
        y = min_y + y * delta_y
        if polygon.contains(Point(x, y)):
            yield (x, y)
