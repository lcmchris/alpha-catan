class A:
    def __init__(self) -> None:
        self.test = self.generate_test()

    def generate_test(self):
        return "test"


class B(A):
    def __init__(self) -> None:
        self.test: int
        super().__init__()

    def generate_test(self):
        print("hello generateedd")
        return 123

    def abc(self):
        return self.test + 2


if __name__ == "__main__":
    b = B()
    print(b.test)
    print(b.abc())
