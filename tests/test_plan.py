import pytest

from dlssnr.plan import KINDS, build_plan, padding, working_extent

SIZES = [(1920, 1080), (2560, 1440), (3840, 2160), (1280, 720), (2048, 1000), (1366, 768), (1000, 1000),
         (3440, 1440), (100, 50), (321, 333)]


@pytest.mark.parametrize("size", SIZES)
def test_layer_count_and_order(size):
    plan = build_plan(*size)
    assert len(plan) == 152
    assert [s.index for s in plan] == list(range(152))
    assert [s.block for s in plan] == sorted(s.block for s in plan)
    assert plan[0].block == 0 and plan[-1].block == 70
    assert {s.kind for s in plan} <= set(KINDS.values())


@pytest.mark.parametrize("size", SIZES)
def test_edges_point_backwards(size):
    plan = build_plan(*size)
    for s in plan:
        assert s.input_block < s.block or (s.input_block == s.block and s.layer > 0) or s.index == 0
        if s.skip_block >= 0:
            assert s.skip_block < s.input_block


@pytest.mark.parametrize("size", SIZES)
def test_extents(size):
    width, height = size
    padded_w, padded_h = working_extent(width, height)
    pad = padding(width, height)
    assert (pad.pad_x, pad.pad_y) == (padded_w - width, padded_h - height)
    assert padded_w >= max(width, 320) and padded_h >= max(height, 320)
    for s in build_plan(width, height):
        assert s.tokens >= s.w * s.h and (s.tokens % 8 == 0 or 31 <= s.block <= 38)
        assert s.grid_x >= 1 and s.grid_y >= 1
        if s.block < 31 or s.block > 39:
            assert s.out_w % 4 == 0 and s.out_h % 4 == 0 or s.kind == "PostBlock"


@pytest.mark.parametrize("size", SIZES)
def test_levels_halve_and_restore(size):
    plan = {(s.block, s.layer): s for s in build_plan(*size)}
    pre = plan[0, 0]
    assert (pre.w, pre.h) == (padding(*size).height, padding(*size).width)
    for down, up in ((4, 66), (8, 62), (14, 56), (22, 48)):
        ds, ups = plan[down, 0], plan[up, 0]
        assert (ups.out_w, ups.out_h) in {(ds.w, ds.h), (_up8(ds.w), _up8(ds.h))}
        assert ups.skip_block == down and ups.input_block == up - 1


def _up8(n):
    return (n + 7) // 8 * 8


def test_variants_and_flags():
    plan = build_plan(1920, 1080)
    by_block = {(s.block, s.layer): s for s in plan}
    assert by_block[0, 0].variant == "ds" and by_block[1, 0].variant == "inpview"
    assert by_block[30, 3].variant == "proj_pool" and by_block[30, 4].variant == "final_head"
    assert by_block[47, 3].variant == "outview" and by_block[66, 0].variant == "upsample"
    assert not by_block[1, 0].shifted and by_block[2, 0].shifted
    assert by_block[2, 0].shift_x == by_block[2, 0].shift_y == -1
    assert by_block[70, 0].skip_block == 0 and by_block[39, 0].skip_block == 30
    assert all(s.shifted == bool(s.shift_x or s.shift_y) for s in plan if s.kind != "PostBlock")


def test_1080p_working_extent_pads_height_to_1088():
    assert working_extent(1920, 1080) == (1920, 1088)


def test_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        build_plan(0, 100)


SMALL_SIZES = [(1920, 1080), (3840, 2160), (1366, 768), (100, 50)]


@pytest.mark.parametrize("size", SMALL_SIZES)
def test_the_small_network_has_the_extracted_records(size):
    """The small plan has a layer for each of the library's weight records (77, with the blend scale) and the same padding."""
    plan = build_plan(*size, "small")
    assert len(plan) == 76 and plan[0].block == 0 and plan[-1].block == 30
    assert [s.block for s in plan] == sorted(s.block for s in plan)
    assert {s.kind for s in plan} <= set(KINDS.values())
    assert padding(*size, "small").width == padding(*size).width     # only the height of 1080 aligns differently


def test_the_small_network_is_shallower_in_the_same_levels():
    plan = {(s.block, s.layer): s for s in build_plan(1920, 1080, "small")}
    assert [plan[b, 0].channels for b in range(1, 9)] == [32, 32, 64, 64, 128, 128, 256, 256]
    assert [plan[b, 0].variant for b in (1, 2, 3, 4)] == ["inpview", "ds", "inpview", "ds"]
    assert [plan[b, 0].kind for b in range(11, 19)] == ["VitFfnExpand"] * 8
    assert plan[10, 4].kind == "SplitFinalHead" and plan[19, 0].kind == "DecInputUpsample"
    assert [plan[b, 0].variant for b in (22, 23, 28, 29)] == ["upsample", "outview", "upsample", "outview"]
    assert plan[19, 0].skip_block == 10 and [plan[b, 0].skip_block for b in (22, 24, 26, 28)] == [8, 6, 4, 2]
    shifts = {b: (plan[b, 2].shift_x, plan[b, 2].shift_y) for b in (9, 10, 20, 21)}
    assert shifts == {9: (0, 0), 10: (-1, -1), 20: (-1, 0), 21: (0, -1)}      # the decoder continues the cycle


def test_an_unknown_network_is_refused():
    with pytest.raises(KeyError):
        build_plan(1920, 1080, "huge")


def test_the_small_network_aligns_an_axis_to_twice_where_the_pre_block_output_is_4_mod_8():
    """Measured against NVIDIA's 310.9.16: 1080 pads to 1152 for the new model, not 1088; the original keeps 1088."""
    assert padding(1920, 1080, "small")[:2] == (1920, 1152) and padding(1920, 1080)[:2] == (1920, 1088)
    assert working_extent(1920, 1080, "small") == (1920, 1152)
    for size in ((2560, 1440), (3840, 2160), (1366, 768), (1280, 720)):       # pre-block outputs 0 mod 8: unchanged
        assert padding(*size, "small") == padding(*size)
    assert padding(1080, 1920, "small")[:2] == (1152, 1920)                    # the rule is per axis
