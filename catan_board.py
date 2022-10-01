
import numpy as np

class CatanBoard():
    def __init__(self):

        self.board = self.generate_board()
    def generate_board():
        arr = np.zeros(10,11)
        for i in range(3,10):
            arr = arr+np.ones(i)
        return arr


    # class Point():
    #     self.



    # class Edges():


    # class Tile(self,resource_type,number):
    #     self.number = number
    #     self.resource_type = resource_type


resource_type = set(['brick','lumber','ore','grain','wool','nothing'])
resource_card = {
    'brick': 19,
    'lumber': 19,
    'ore':19,
    'grain':19,
    'wool':19
}
knight_card = {
    'knight': 14,
    'victory_point': 5,
    'road_building':2,
    'year_of_plenty':2,
    'monopoly': 2
}

if __name__ == 'main':
    print(CatanBoard.board)
