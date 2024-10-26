def HorizontalRegion(x: float) -> str:
    if x < 0.033:
        return "Right"
    elif x > 0.037:
        return "Left"
    else:
        return "Center"

def VerticalRegion(y: float) -> str:
    if y > 0.073:
        return "Up"
    elif y < 0.069:
        return "Down"
    else:
        return "Center"