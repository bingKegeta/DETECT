function HorizontalRegion(x) {
    if (x < 0.033) {
        return "Right";
    } else if (x > 0.037) {
        return "Left";
    } else {
        return "Center";
    }
}

function VerticalRegion(y) {
    if (y > 0.073) {
        return "Up";
    } else if (y < 0.069) {
        return "Down";
    } else {
        return "Center";
    }
}

module.exports = { HorizontalRegion, VerticalRegion };
