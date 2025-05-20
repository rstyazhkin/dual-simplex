import numpy as np

class Simplex():
    def __init__(self, A, P, c, signs):
        self.A = np.array(A, dtype=float)               # Коэффициенты ограничений
        self.P = np.array(P, dtype=float)               # Свободные члены
        self.c = np.array(c, dtype=float)               # Коэффициенты целевой функции
        self.signs = signs                              # Знаки ограничений (1 - <=; 2 - >=; 3 - =)
        self.restr_num, self.var_num = self.A.shape
        self.B = []                                     # Базисы
        self.variable_rows = None
        self.function_row = None
        self.func_val = None           
        self.tableu = None
        
    def _initialize_table(self):
        '''
        Приводит задачу к нормальному виду, а также инициализирует начальную симплекс таблицу
        '''
        A = self.A
        P = self.P
        c = self.c
        signs = self.signs
        
        slack_num, artificial_num = self._count_additional_variables()
        slack_idx = self.var_num
        artificial_idx = slack_idx + slack_num
        M = 10**6
        
        for restr in range(self.restr_num):
            if P[restr] < 0:
                A[restr] = [-x for x in A[restr]]
                P[restr] *= -1
                if signs[restr] == 1:
                    signs[restr] = 2
                elif signs[restr] == 2:
                    signs[restr] = 1
        
        # Массив коэффициентов ограничений
        variable_rows = [list(x) + [0] * (slack_num + artificial_num) for x in A]
        # Массив коэффициентов целевой функции
        function_row = list(c) + [0] * (slack_num + artificial_num)
        
        for restr in range(self.restr_num):                    
            # Добавить слэки
            if signs[restr] == 1:
                variable_rows[restr][slack_idx] = 1
                self.B.append(slack_idx)
                slack_idx += 1
            elif signs[restr] == 2:
                variable_rows[restr][slack_idx] = -1
                variable_rows[restr][artificial_idx] = 1
                function_row[artificial_idx] = -M
                self.B.append(artificial_idx)
                slack_idx += 1
                artificial_idx += 1
            else:
                variable_rows[restr][artificial_idx] = 1
                function_row[artificial_idx] = -M
                self.B.append(artificial_idx)
                artificial_idx += 1
    
        Cb = np.array([function_row[x] for x in self.B], dtype=float)
        self.func_val = sum([x * y for x, y in zip(Cb, P)])
        variable_rows = np.array(variable_rows, dtype=float)
        z_row = (Cb @ variable_rows - function_row).tolist()
        
        self.tableu = np.vstack((variable_rows, z_row))
        self.function_row = function_row
        
        # entering = z_row.index(min(z_row))     # Переменная, входящая в базис | индекс ведущего столбца
        # pivot_col = variable_rows[:, entering].tolist()
        
        # Q = [x / y for x, y in zip(P, pivot_col)]
        # leaving = self.B[Q.index(min(filter(lambda x: x > 0, Q)))]
        # pivot_row = variable_rows[Q.index(min(filter(lambda x: x > 0, Q))), :]
        # Q = np.array(Q, dtype=float)
        # print(variable_rows)
        # print(f"Входит в базис: {entering}")
        # print(f"Выходит из базиса: {leaving}")
        # print(f"Ведущий столбец: {pivot_col}")
        # print(f"Ведущая строка: {pivot_row}")
        for i, basis in enumerate(self.B):
            print(f"Базис {i + 1}: {basis + 1}")
        print("Инициализирована симплекс-таблица:")
        for row in self.tableu:
            print(" ".join(f"{x:^12.2f}" for x in row))
        
        self._define_pivot(self.B, self.tableu)
                    
    def _count_additional_variables(self):
        slack_number = 0
        artifitial_number = 0
        
        for restr in range(self.restr_num):
            if self.signs[restr] == 1:
               slack_number += 1
            elif self.signs[restr] == 2:
                slack_number += 1
                artifitial_number += 1
            else:
                artifitial_number += 1
        
        return (slack_number, artifitial_number)
    
    def _define_pivot(self, B, tableau):
        z_row = tableau[-1,:].tolist()
        variable_rows = tableau[:-1,:]
        
        entering = z_row.index(min(z_row))     # Переменная, входящая в базис | индекс ведущего столбца
        pivot_col = variable_rows[:, entering].tolist()
        
        try:
            Q = [x / y for x, y in zip(P, pivot_col)]
        except ZeroDivisionError:
            print("Задача неограничена")
            
        leaving =  B[Q.index(min(filter(lambda x: x > 0, Q)))]
        # pivot_row = variable_rows[Q.index(min(filter(lambda x: x > 0, Q))), :]
        pivot_element = tableau[Q.index(min(filter(lambda x: x > 0, Q))), entering]
        # print(entering)
        # print(leaving)
        # print(pivot_element)
        return (entering, leaving, pivot_element)
        
    def _is_solved(self, tableau):
        z_row = tableau[-1,:].tolist()
        if all([x > 0 for x in z_row]):
            return True
        else:
            return False
    
    def _iterate(self, tableau):
        entering, leaving, pivot_element = self._define_pivot(self.B, tableau)
        B_new = self.B.copy()
        leaving_idx = B_new.index(leaving)
        B_new[leaving_idx] = entering
        self.B = B_new  # обновляем базис

        # Шаг 1: Нормализуем ведущую строку
        new_tableau = tableau.copy()
        pivot_row_index = leaving_idx
        pivot_column_index = entering

        new_tableau[pivot_row_index] = new_tableau[pivot_row_index] / pivot_element

        # Шаг 2: Обнуляем все остальные элементы в ведущем столбце (кроме ведущей строки)
        for i in range(len(new_tableau)):
            if i != pivot_row_index:
                factor = new_tableau[i, pivot_column_index]
                new_tableau[i] = new_tableau[i] - factor * new_tableau[pivot_row_index]

        # Шаг 3: Обновляем строку z
        Cb = np.array([self.function_row[x] for x in self.B], dtype=float)
        variable_rows = new_tableau[:-1, :]
        z_row = (Cb @ variable_rows - self.function_row).tolist()
        new_tableau[-1, :] = z_row
        self.func_val = Cb @ self.P

        # Обновляем таблицу
        self.tableu = new_tableau

        print("\nПосле итерации симплекс-таблица:")
        for row in self.tableu:
            print(" ".join(f"{x:^12.2f}" for x in row))
        print(f"Текущий базис: {[x + 1 for x in self.B]}")

        
    
    def solve(self):
        # Приведение задачи к каноническому виду
        # Инициализация симплекс таблицы
        self._initialize_table()
        
        # Итерационный процесс
        # while not self._is_solved():
        #     solution = self._iterate()
        self._iterate(self.tableu)
        # print("Решение найдено:")
        
    
if __name__ == "__main__":
    A = [
        [3, 7, 2, 4],
        [8, 4, 4, 2],
        [14, 3, 22, -30],
        [5, 3, 5, 3]
    ]
    P = [4, -10, 31, -3] 
    c = [5, 20, 1, -30]
    signs = [3, 2, 1, 1]
    simplex = Simplex(A, P, c, signs)
    simplex.solve()