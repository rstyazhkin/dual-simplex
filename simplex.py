import numpy as np

class Simplex():
    def __init__(self, A, P, c, signs):
        self.A = np.array(A, dtype=float)
        self.P = np.array(P, dtype=float)
        self.c = np.array(c, dtype=float)
        self.signs = signs
        self.restr_num, self.var_num = self.A.shape
        self.B = []
        self.function_row = None
        self.func_val = None
        self.tableu = None
        
    def _initialize_table(self):
        A, P, c, signs = self.A, self.P, self.c, self.signs
        slack_idx = self.var_num
        M = 10**6

        for i in range(self.restr_num):
            if P[i] < 0:
                A[i] = -A[i]
                P[i] *= -1
                if signs[i] == 1:
                    signs[i] = 2
                elif signs[i] == 2:
                    signs[i] = 1

        slack_num, artificial_num = self._count_additional_variables()
        artificial_idx = slack_idx + slack_num
        
        variable_rows = [list(row) + [0]*(slack_num + artificial_num) for row in A]
        function_row = list(c) + [0]*(slack_num + artificial_num)

        for i in range(self.restr_num):
            if signs[i] == 1:
                variable_rows[i][slack_idx] = 1
                self.B.append(slack_idx)
                slack_idx += 1
            elif signs[i] == 2:
                variable_rows[i][slack_idx] = -1
                variable_rows[i][artificial_idx] = 1
                function_row[artificial_idx] = -M
                self.B.append(artificial_idx)
                slack_idx += 1
                artificial_idx += 1
            else:
                variable_rows[i][artificial_idx] = 1
                function_row[artificial_idx] = -M
                self.B.append(artificial_idx)
                artificial_idx += 1

        variable_rows = np.array(variable_rows, dtype=float)
        Cb = np.array([function_row[x] for x in self.B], dtype=float)
        z_row = (Cb @ variable_rows - function_row).tolist()

        self.tableu = np.vstack((variable_rows, z_row))
        self.function_row = function_row
        self.func_val = Cb @ P

        print("Инициализирована симплекс-таблица:")
        for row in self.tableu:
            print(" ".join(f"{x:^12.2f}" for x in row))
        print("P:", np.round(self.P, 3))
        print(f"Начальный базис: {[x + 1 for x in self.B]}")

    def _count_additional_variables(self):
        slack, artif = 0, 0
        for sign in self.signs:
            if sign == 1:
                slack += 1
            elif sign == 2:
                slack += 1
                artif += 1
            else:
                artif += 1
        return slack, artif

    def _define_pivot(self, B, tableau, P):
        z_row = tableau[-1, :]
        variable_rows = tableau[:-1, :]
        entering = int(np.argmin(z_row))
        pivot_col = variable_rows[:, entering]

        Q = [p / a if a > 0 else float('inf') for p, a in zip(P, pivot_col)]
        if all(q == float('inf') for q in Q):
            raise ValueError("Задача неограничена — нет допустимого Q-отношения")

        pivot_row_index = int(np.argmin(Q))
        pivot_element = tableau[pivot_row_index, entering]
        leaving = B[pivot_row_index]
        return entering, leaving, pivot_element, pivot_row_index

    def _is_solved(self, tableau):
        z_row = tableau[-1, :]
        return all(x >= -1e-8 for x in z_row)

    def _iterate(self, tableau):
        entering, leaving, pivot_element, pivot_row_index = self._define_pivot(self.B, tableau, self.P)
        pivot_col_idx = entering

        self.B[pivot_row_index] = entering

        new_tableau = tableau.copy()
        new_P = self.P.copy()

        new_tableau[pivot_row_index] /= pivot_element
        new_P[pivot_row_index] /= pivot_element

        for i in range(len(new_tableau) - 1):
            if i != pivot_row_index:
                factor = new_tableau[i, pivot_col_idx]
                new_tableau[i] -= factor * new_tableau[pivot_row_index]
                new_P[i] -= factor * new_P[pivot_row_index]

        Cb = np.array([self.function_row[x] for x in self.B], dtype=float)
        z_row = Cb @ new_tableau[:-1, :] - self.function_row
        new_tableau[-1, :] = z_row
        self.func_val = Cb @ new_P

        self.tableu = new_tableau
        self.P = new_P

        for row in self.tableu:
            print(" ".join(f"{x:^12.2f}" for x in row))
        print("P:", np.round(self.P, 3))
        print(f"Базис: {[x + 1 for x in self.B]}")
        print(f"Текущее значение функции: {np.round(self.func_val, 3)}")

    def solve(self):
        self._initialize_table()
        iteration = 1
        while not self._is_solved(self.tableu):
            print()
            print(f"Итерация {iteration}")
            self._iterate(self.tableu)
            iteration += 1

        print("\nРешение найдено!")
        solution = [0] * self.var_num
        for i, b in enumerate(self.B):
            if b < self.var_num:
                solution[b] = round(self.P[i], 6)

        for idx, val in enumerate(solution):
            print(f"x{idx + 1} = {val}")

        print(f"Оптимальное значение функции: {round(self.func_val, 6)}")


def parse_input_file(filename="input.txt"):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    section = None
    c = []
    A = []
    signs = []
    P = []

    for line in lines:
        lower = line.lower()
        if "function coefficients" in lower:
            section = "function"
        elif "restriction coefficients" in lower:
            section = "restrictions"
        elif "restriction signs" in lower:
            section = "signs"
        elif "restriction right-side values" in lower:
            section = "rhs"
        else:
            nums = line.split()
            if section == "function":
                c = list(map(float, nums))
            elif section == "restrictions":
                A.append(list(map(float, nums)))
            elif section == "signs":
                # Преобразуем знаки в: <= → 1, >= → 2, = → 3
                mapping = {"<=": 1, ">=": 2, "=": 3}
                signs = [mapping[s] for s in nums]
            elif section == "rhs":
                P = list(map(float, nums))

    return A, P, c, signs

# Пример использования
if __name__ == "__main__":
    A, P, c, signs = parse_input_file()
    simplex = Simplex(A, P, c, signs)
    simplex.solve()
