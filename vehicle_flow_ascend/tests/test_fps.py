from vehicle_flow_ascend.utils.fps import FpsMeter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_fps_is_zero_before_two_ticks() -> None:
    clock = FakeClock()
    meter = FpsMeter(time_fn=clock)

    assert meter.fps == 0.0
    assert meter.tick() == 0.0


def test_fps_uses_elapsed_time_between_ticks() -> None:
    clock = FakeClock()
    meter = FpsMeter(time_fn=clock)

    meter.tick()
    clock.advance(0.5)
    fps = meter.tick()

    assert fps == 2.0


def test_fps_uses_rolling_window() -> None:
    clock = FakeClock()
    meter = FpsMeter(window_size=3, time_fn=clock)

    meter.tick()
    clock.advance(1.0)
    meter.tick()
    clock.advance(1.0)
    meter.tick()
    clock.advance(1.0)
    fps = meter.tick()

    assert fps == 1.0
