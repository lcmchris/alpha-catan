from catan_game import Catan
import pytest
import numpy as np


@pytest.fixture
def catan():
    yield Catan(seed=1, player_type=["random"], player_count=1)


def test_longest_road(catan: Catan):
    # fmt: off
    catan.board = np.array(  
        []
    )
    # fmt: on

    assert catan.players[-9].calculate_longest_road() == 1
