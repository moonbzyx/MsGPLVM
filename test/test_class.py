import torch


class A():
    # the level of variable 'b' is equal to function
    # it's the attribute of class A, that is: self.b
    b = 6

    def __init__(self):
        self.a = 5

    def test(self):
        a = 7
        b = 8
        print('a', a)
        print('self.a', self.a)
        print('b', b)
        print('self.b', self.b)


test = A()
test.test()