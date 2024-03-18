import time

class Timer:
    def start(self, task="0.0"):
        self.task = task
        self.t1 = time.time()
        print(f"\nTask {self.task}")

    def end(self):
        self.t2 = time.time()
        print(f'Task {self.task} time: {self.t2 - self.t1:.4e}s')


# def Task(task_function):
#     print(f'Inside')
#     def wrapper(exec_function):
#         # func.__name__,
#         # t1 = time.time()
#         print(f'{exec_function = }')

#         return task_function()

#     return wrapper