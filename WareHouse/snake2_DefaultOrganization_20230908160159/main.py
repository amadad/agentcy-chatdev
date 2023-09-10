'''
Classic Snake Game
'''
import tkinter as tk
import random
class SnakeGame:
    def __init__(self, window):
        self.window = window
        self.window.title("Snake Game")
        self.canvas = tk.Canvas(window, width=400, height=400, bg="black")
        self.canvas.pack()
        self.window.bind("<Key>", self.on_key_press)
        self.direction = "Right"
        self.snake = [(200, 200), (190, 200), (180, 200)]
        self.food = self.create_food()
        self.score = 0
        self.speed = 100
        self.game_over = False
        self.move_snake()
    def create_food(self):
        x = random.randint(0, 39) * 10
        y = random.randint(0, 39) * 10
        return self.canvas.create_oval(x, y, x + 10, y + 10, fill="red")
    def move_snake(self):
        if not self.game_over:
            head_x, head_y = self.snake[0]
            if self.direction == "Up":
                head_y -= 10
            elif self.direction == "Down":
                head_y += 10
            elif self.direction == "Left":
                head_x -= 10
            elif self.direction == "Right":
                head_x += 10
            self.snake.insert(0, (head_x, head_y))
            self.canvas.delete("snake")
            for segment in self.snake:
                x, y = segment
                self.canvas.create_rectangle(x, y, x + 10, y + 10, fill="green", tags="snake")
            if head_x == self.canvas.coords(self.food)[0] and head_y == self.canvas.coords(self.food)[1]:
                self.score += 1
                self.canvas.delete(self.food)
                self.food = self.create_food()
                self.speed -= 2
            else:
                self.snake.pop()
            if (
                head_x < 0
                or head_x >= 400
                or head_y < 0
                or head_y >= 400
                or (head_x, head_y) in self.snake[1:]
            ):
                self.game_over = True
            self.canvas.after(self.speed, self.move_snake)
    def on_key_press(self, event):
        if event.keysym == "Up" and self.direction != "Down":
            self.direction = "Up"
        elif event.keysym == "Down" and self.direction != "Up":
            self.direction = "Down"
        elif event.keysym == "Left" and self.direction != "Right":
            self.direction = "Left"
        elif event.keysym == "Right" and self.direction != "Left":
            self.direction = "Right"
if __name__ == "__main__":
    window = tk.Tk()
    game = SnakeGame(window)
    window.mainloop()