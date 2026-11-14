import random
from pathlib import Path

import joblib
import pandas as pd
from pygame import *

from collect_data import DataCollector

init()

# Розміри вікна та сітки
WIDTH = 640
HEIGHT = 650
FPS = 60

COLS = 10
ROWS = 20
BLOCK_SIZE = 26

BOARD_X = 40
BOARD_Y = 50
BOARD_WIDTH = COLS * BLOCK_SIZE
BOARD_HEIGHT = ROWS * BLOCK_SIZE

screen = display.set_mode((WIDTH, HEIGHT))
display.set_caption("Simple Tetris with AI")

clock = time.Clock()
title_font = font.Font(None, 42)
game_font = font.Font(None, 34)
info_font = font.Font(None, 24)
help_font = font.Font(None, 20)

# Кольори фігур (RGB)
COLORS = {
    "I": (0, 220, 220),    # Бірюзовий
    "O": (240, 210, 0),    # Жовтий
    "T": (170, 60, 230),   # Фіолетовий
    "S": (60, 210, 80),    # Зелений
    "Z": (235, 65, 65),    # Червоний
    "J": (40, 110, 240),   # Синій
    "L": (240, 140, 30),   # Помаранчевий
}

# Визначення форм і обертань через відносні координати (dx, dy)
SHAPES = {
    "I": [
        [(-1, 0), (0, 0), (1, 0), (2, 0)],
        [(0, -1), (0, 0), (0, 1), (0, 2)],
        [(-1, 0), (0, 0), (1, 0), (2, 0)],
        [(0, -1), (0, 0), (0, 1), (0, 2)],
    ],
    "O": [
        [(0, 0), (1, 0), (0, 1), (1, 1)],
    ],
    "T": [
        [(-1, 0), (0, 0), (1, 0), (0, -1)],
        [(0, -1), (0, 0), (0, 1), (1, 0)],
        [(-1, 0), (0, 0), (1, 0), (0, 1)],
        [(0, -1), (0, 0), (0, 1), (-1, 0)],
    ],
    "S": [
        [(0, 0), (1, 0), (0, 1), (-1, 1)],
        [(0, -1), (0, 0), (1, 0), (1, 1)],
    ],
    "Z": [
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
        [(1, -1), (1, 0), (0, 0), (0, 1)],
    ],
    "J": [
        [(-1, -1), (-1, 0), (0, 0), (1, 0)],
        [(0, -1), (1, -1), (0, 0), (0, 1)],
        [(-1, 0), (0, 0), (1, 0), (1, 1)],
        [(0, -1), (0, 0), (0, 1), (-1, 1)],
    ],
    "L": [
        [(1, -1), (-1, 0), (0, 0), (1, 0)],
        [(0, -1), (0, 0), (0, 1), (1, 1)],
        [(-1, 0), (0, 0), (1, 0), (-1, 1)],
        [(-1, -1), (0, -1), (0, 0), (0, 1)],
    ],
}

# Базові 7 стандартних фігур
STANDARD_SHAPES = ["I", "O", "T", "S", "Z", "J", "L"]

# НОВЕ: додаткові рандомно створені нестандартні фігури, що активуються після 250 очок
CUSTOM_SHAPES = {
    "DOT": [
        [(0, 0)],
    ],
    "CORNER": [
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (0, 1), (-1, 0)],
        [(0, 0), (-1, 0), (0, -1)],
        [(0, 0), (0, -1), (1, 0)],
    ],
    "PLUS": [
        [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)],
    ],
    "BAR3": [
        [(-1, 0), (0, 0), (1, 0)],
        [(0, -1), (0, 0), (0, 1)],
    ],
}

# Додаємо кастомні форми та їхні кольори
SHAPES.update(CUSTOM_SHAPES)
COLORS.update({
    "DOT": (240, 240, 255),     # Білий кристал
    "CORNER": (255, 105, 180),  # Рожевий
    "PLUS": (255, 75, 75),      # Яскраво-червоний
    "BAR3": (70, 220, 130),     # Лаймовий
})

# НОВЕ: параметри підвищеної складності
HARD_MODE_SCORE = 250
OBSTACLE_COLOR = (145, 145, 155)  # Колір блоку-перешкоди (металевий сірий з червоним маркуванням)


def spawn_random_board_block(board):
    """Створює випадковий блок-перешкоду на полі (після 250 очок)."""
    empty_cells = [
        (r, c)
        for r in range(8, ROWS)
        for c in range(COLS)
        if board[r][c] == 0
    ]
    if empty_cells:
        r, c = random.choice(empty_cells)
        board[r][c] = OBSTACLE_COLOR
        print(f"⚠️ Спавн блоку-перешкоди в клітинці ({r}, {c})")
        return (r, c)
    return None


def get_available_shapes(current_score):
    """Повертає доступні фігури: після 250 очок додаються рандомно створені форми."""
    if current_score >= HARD_MODE_SCORE:
        return STANDARD_SHAPES + list(CUSTOM_SHAPES.keys())
    return STANDARD_SHAPES


# Створюємо збирач даних
collector = DataCollector("tetris_data.csv")

# Шлях до навченої моделі
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(".").resolve()

MODEL_FILE = BASE_DIR / "tetris_model.joblib"

# Ознаки, які використовувались під час навчання
DEFAULT_FEATURE_COLUMNS = [
    "lines_cleared",
    "holes",
    "bumpiness",
    "aggregate_height",
    "max_height",
    "landing_height",
]


def load_ai_model():
    """Завантаження навченої моделі."""
    if not MODEL_FILE.exists():
        print("Модель не знайдено:", MODEL_FILE)
        print("Гра працюватиме в ручному режимі або з базовим алгоритмом.")
        return None, DEFAULT_FEATURE_COLUMNS

    try:
        model_data = joblib.load(MODEL_FILE)
        if isinstance(model_data, dict):
            loaded_model = model_data["model"]
            feature_columns = model_data.get(
                "feature_columns",
                DEFAULT_FEATURE_COLUMNS,
            )
            return loaded_model, feature_columns
        return model_data, DEFAULT_FEATURE_COLUMNS
    except Exception as e:
        print("Помилка завантаження моделі:", e)
        return None, DEFAULT_FEATURE_COLUMNS


ai_model, ai_feature_columns = load_ai_model()

# Початковий режим керування — гравець керує власноруч (HUMAN)
control_mode = "human"

last_ai_action = "none"


# Функції роботи з полем та метриками
def create_empty_board():
    return [[0 for _ in range(COLS)] for _ in range(ROWS)]


def get_column_heights(board):
    heights = [0] * COLS
    for col in range(COLS):
        for row in range(ROWS):
            if board[row][col] != 0:
                heights[col] = ROWS - row
                break
    return heights


def calculate_board_metrics(board, lines_cleared, landing_height):
    heights = get_column_heights(board)
    aggregate_height = sum(heights)
    max_height = max(heights) if heights else 0

    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(COLS - 1))

    holes = 0
    for col in range(COLS):
        seen_block = False
        for row in range(ROWS):
            if board[row][col] != 0:
                seen_block = True
            elif seen_block:
                holes += 1

    return {
        "lines_cleared": lines_cleared,
        "holes": holes,
        "bumpiness": bumpiness,
        "aggregate_height": aggregate_height,
        "max_height": max_height,
        "landing_height": landing_height,
    }


def is_valid_position(board, shape, rot, px, py):
    num_rots = len(SHAPES[shape])
    offsets = SHAPES[shape][rot % num_rots]
    for dx, dy in offsets:
        x = px + dx
        y = py + dy
        if x < 0 or x >= COLS:
            return False
        if y >= ROWS:
            return False
        if y >= 0 and board[y][x] != 0:
            return False
    return True


def get_drop_y(board, shape, rot, px, py):
    cur_y = py
    while is_valid_position(board, shape, rot, px, cur_y + 1):
        cur_y += 1
    return cur_y


def clear_lines(board):
    """Поставлені блоки залишаються на полі і знищуються ТІЛЬКИ тоді,
    коли вони вибудовуються у один повністю заповнений рядок (усі 10 клітинок).
    Після знищення заповненого рядка всі блоки вище опускаються вниз."""
    new_board = []
    cleared = 0
    for r in range(ROWS):
        # Рядок повністю заповнений ТІЛЬКИ якщо всі 10 клітинок непорожні (!= 0)
        is_full_row = all(board[r][c] != 0 for c in range(COLS))
        if is_full_row:
            cleared += 1
        else:
            new_board.append(board[r][:])

    # Заповнюємо звільнені зверху рядки новими порожніми
    for _ in range(cleared):
        new_board.insert(0, [0] * COLS)

    return new_board, cleared


def simulate_placement(board, shape, rot, px, py):
    temp_board = [row[:] for row in board]
    for dx, dy in SHAPES[shape][rot]:
        x = px + dx
        y = py + dy
        if 0 <= y < ROWS and 0 <= x < COLS:
            temp_board[y][x] = 1

    temp_board, lines_cleared = clear_lines(temp_board)
    landing_height = ROWS - py
    metrics = calculate_board_metrics(temp_board, lines_cleared, landing_height)
    return temp_board, metrics


def evaluate_placement(model, feature_cols, metrics):
    prob_good = 0.5
    if model is not None:
        try:
            state_values = {col: metrics.get(col, 0) for col in feature_cols}
            df = pd.DataFrame([state_values])
            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                proba = model.predict_proba(df)[0]
                if "good" in classes:
                    prob_good = proba[classes.index("good")]
                else:
                    prob_good = proba[-1]
            else:
                pred = model.predict(df)[0]
                prob_good = 1.0 if pred == "good" else 0.0
        except Exception:
            prob_good = 0.5

    # Комбінована функція корисності: враховує ймовірність моделі та класичні правила Тетрісу
    score = (
        prob_good * 400.0
        + metrics["lines_cleared"] * 160.0
        - metrics["holes"] * 80.0
        - metrics["bumpiness"] * 6.0
        - metrics["aggregate_height"] * 3.0
        - metrics["landing_height"] * 2.0
    )
    return score


def get_best_ai_placement(board, shape, model, feature_cols):
    best_placement = None
    best_score = -float("inf")
    num_rots = len(SHAPES[shape])

    for rot in range(num_rots):
        offsets = SHAPES[shape][rot]
        min_dx = min(dx for dx, dy in offsets)
        max_dx = max(dx for dx, dy in offsets)
        min_x = -min_dx
        max_x = COLS - 1 - max_dx

        for x in range(min_x, max_x + 1):
            if is_valid_position(board, shape, rot, x, 0):
                drop_y = get_drop_y(board, shape, rot, x, 0)
                _, metrics = simulate_placement(board, shape, rot, x, drop_y)
                score = evaluate_placement(model, feature_cols, metrics)
                if score > best_score:
                    best_score = score
                    best_placement = (rot, x, drop_y)

    return best_placement


# Стан гри
board = create_empty_board()
score = 0
lines = 0
game_over = False

init_shapes = get_available_shapes(score)
shape_list = init_shapes
current_shape = random.choice(init_shapes)
next_shape = random.choice(init_shapes)

current_rot = 0
current_x = COLS // 2
current_y = 0

fall_speed = 30  # кадрів на одне падіння в Human режимі
fall_counter = 0

ai_target_rot = 0
ai_target_x = current_x
ai_move_interval = 4  # швидкість кроків AI
ai_move_counter = 0


def reset_game():
    global board, score, lines, game_over, current_shape, next_shape
    global current_rot, current_x, current_y, fall_counter, ai_move_counter
    board = create_empty_board()
    score = 0
    lines = 0
    game_over = False
    init_shapes = get_available_shapes(score)
    current_shape = random.choice(init_shapes)
    next_shape = random.choice(init_shapes)
    current_rot = 0
    current_x = COLS // 2
    current_y = 0
    fall_counter = 0
    ai_move_counter = 0
    plan_ai_move()


def plan_ai_move():
    global ai_target_rot, ai_target_x
    placement = get_best_ai_placement(
        board,
        current_shape,
        ai_model,
        ai_feature_columns,
    )
    if placement:
        ai_target_rot, ai_target_x, _ = placement
    else:
        ai_target_rot = 0
        ai_target_x = current_x


plan_ai_move()

running = True

while running:
    clock.tick(FPS)

    for current_event in event.get():
        if current_event.type == QUIT:
            running = False

        if current_event.type == KEYDOWN:
            # Клавіша M перемикає режим керування
            if current_event.key == K_m:
                if control_mode == "human":
                    control_mode = "ai"
                    plan_ai_move()
                    print("Режим керування: AI")
                else:
                    control_mode = "human"
                    print("Режим керування: HUMAN")

            # Перезапуск гри
            if current_event.key == K_r:
                reset_game()

            # Ручне керування гравцем
            if control_mode == "human" and not game_over:
                if current_event.key == K_LEFT:
                    if is_valid_position(board, current_shape, current_rot, current_x - 1, current_y):
                        current_x -= 1
                elif current_event.key == K_RIGHT:
                    if is_valid_position(board, current_shape, current_rot, current_x + 1, current_y):
                        current_x += 1
                elif current_event.key == K_UP:
                    new_rot = (current_rot + 1) % len(SHAPES[current_shape])
                    # Перевіряємо обертання з можливими зміщеннями від стіни (wall kick)
                    for kick_x in [0, -1, 1, -2, 2]:
                        if is_valid_position(board, current_shape, new_rot, current_x + kick_x, current_y):
                            current_rot = new_rot
                            current_x += kick_x
                            break
                elif current_event.key == K_DOWN:
                    if is_valid_position(board, current_shape, current_rot, current_x, current_y + 1):
                        current_y += 1
                        score += 1
                elif current_event.key == K_SPACE:
                    # Hard drop
                    drop_y = get_drop_y(board, current_shape, current_rot, current_x, current_y)
                    score += (drop_y - current_y) * 2
                    current_y = drop_y
                    fall_counter = fall_speed

    # Логіка оновлення гри
    if not game_over:
        # Режим AI
        if control_mode == "ai":
            ai_move_counter += 1
            if ai_move_counter >= ai_move_interval:
                ai_move_counter = 0

                num_rots = len(SHAPES[current_shape])
                target_rot = ai_target_rot % num_rots

                if current_rot != target_rot:
                    new_rot = (current_rot + 1) % num_rots
                    if is_valid_position(board, current_shape, new_rot, current_x, current_y):
                        current_rot = new_rot
                        last_ai_action = "rotate"
                    else:
                        # Спроба зі зміщенням
                        for kick_x in [-1, 1, -2, 2]:
                            if is_valid_position(board, current_shape, new_rot, current_x + kick_x, current_y):
                                current_rot = new_rot
                                current_x += kick_x
                                last_ai_action = "rotate"
                                break
                elif current_x < ai_target_x:
                    if is_valid_position(board, current_shape, current_rot, current_x + 1, current_y):
                        current_x += 1
                        last_ai_action = "right"
                elif current_x > ai_target_x:
                    if is_valid_position(board, current_shape, current_rot, current_x - 1, current_y):
                        current_x -= 1
                        last_ai_action = "left"
                else:
                    # Фігура на потрібній позиції — опускаємо
                    last_ai_action = "drop"
                    drop_y = get_drop_y(board, current_shape, current_rot, current_x, current_y)
                    current_y = drop_y
                    fall_counter = fall_speed

        # Природне падіння фігури
        fall_counter += 1
        if fall_counter >= fall_speed:
            fall_counter = 0

            if is_valid_position(board, current_shape, current_rot, current_x, current_y + 1):
                current_y += 1
            else:
                # Фіксація фігури на полі
                color_val = COLORS[current_shape]
                for dx, dy in SHAPES[current_shape][current_rot]:
                    x = current_x + dx
                    y = current_y + dy
                    if 0 <= y < ROWS and 0 <= x < COLS:
                        board[y][x] = color_val

                # Очищення ліній: знищуються ТІЛЬКИ повністю заповнені рядки
                board, cleared = clear_lines(board)
                if cleared > 0:
                    lines += cleared
                    points = [0, 100, 300, 500, 800]
                    score += points[min(cleared, 4)]
                    print(f"🎉 Знищено {cleared} повністю заповнений рядок! (Всього ліній: {lines}, Рахунок: {score})")

                # НОВЕ: після 250 очок спавниться випадковий блок-перешкода на полі
                if score >= HARD_MODE_SCORE:
                    spawn_random_board_block(board)

                landing_height = ROWS - current_y
                metrics = calculate_board_metrics(board, cleared, landing_height)

                # Визначаємо дію для збереження в датасет
                action_label = "good" if (cleared > 0 or metrics["holes"] == 0) else "bad"

                collector.collect(
                    lines_cleared=metrics["lines_cleared"],
                    holes=metrics["holes"],
                    bumpiness=metrics["bumpiness"],
                    aggregate_height=metrics["aggregate_height"],
                    max_height=metrics["max_height"],
                    landing_height=metrics["landing_height"],
                    score=score,
                    action=action_label,
                )

                # Створюємо наступну фігуру (після 250 очок можуть бути кастомні фігури)
                current_shape = next_shape
                available_shapes = get_available_shapes(score)
                shape_list = available_shapes
                next_shape = random.choice(available_shapes)
                current_rot = 0
                current_x = COLS // 2
                current_y = 0

                if not is_valid_position(board, current_shape, current_rot, current_x, current_y):
                    game_over = True
                    print("Game Over!")

                if control_mode == "ai" and not game_over:
                    plan_ai_move()

    # ВІДОБРАЖЕННЯ
    screen.fill((22, 25, 34))

    # Рамка та фон ігрового поля
    board_rect = Rect(BOARD_X, BOARD_Y, BOARD_WIDTH, BOARD_HEIGHT)
    draw.rect(screen, (15, 17, 23), board_rect)
    draw.rect(screen, (70, 75, 95), board_rect, 2)

    # Сітка поля
    for r in range(ROWS):
        for c in range(COLS):
            cell_rect = Rect(
                BOARD_X + c * BLOCK_SIZE,
                BOARD_Y + r * BLOCK_SIZE,
                BLOCK_SIZE,
                BLOCK_SIZE,
            )
            draw.rect(screen, (28, 32, 42), cell_rect, 1)

            cell_val = board[r][c]
            if cell_val != 0:
                draw.rect(screen, cell_val, cell_rect)
                if cell_val == OBSTACLE_COLOR:
                    # Червона рамка та діагональний хрестик для блоку-перешкоди
                    draw.rect(screen, (230, 60, 60), cell_rect, 2)
                    draw.line(screen, (230, 60, 60), (cell_rect.left + 4, cell_rect.top + 4), (cell_rect.right - 5, cell_rect.bottom - 5), 2)
                    draw.line(screen, (230, 60, 60), (cell_rect.left + 4, cell_rect.bottom - 5), (cell_rect.right - 5, cell_rect.top + 4), 2)
                else:
                    draw.rect(screen, (255, 255, 255), cell_rect, 1)

    # Привидна фігура (Ghost Piece)
    if not game_over:
        ghost_y = get_drop_y(board, current_shape, current_rot, current_x, current_y)
        for dx, dy in SHAPES[current_shape][current_rot]:
            gx = current_x + dx
            gy = ghost_y + dy
            if 0 <= gy < ROWS and 0 <= gx < COLS:
                ghost_rect = Rect(
                    BOARD_X + gx * BLOCK_SIZE,
                    BOARD_Y + gy * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                )
                draw.rect(screen, (60, 65, 80), ghost_rect, 1)

        # Поточна активна фігура
        current_color = COLORS[current_shape]
        for dx, dy in SHAPES[current_shape][current_rot]:
            px = current_x + dx
            py = current_y + dy
            if 0 <= py < ROWS and 0 <= px < COLS:
                piece_rect = Rect(
                    BOARD_X + px * BLOCK_SIZE,
                    BOARD_Y + py * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                )
                draw.rect(screen, current_color, piece_rect)
                draw.rect(screen, (255, 255, 255), piece_rect, 1)

    # ПРАВА ПАНЕЛЬ: статистика та інформація
    info_x = BOARD_X + BOARD_WIDTH + 35

    title_text = title_font.render("TETRIS AI", True, (255, 220, 50))
    screen.blit(title_text, (info_x, 40))

    score_text = game_font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (info_x, 95))

    lines_text = game_font.render(f"Lines: {lines}", True, (255, 255, 255))
    screen.blit(lines_text, (info_x, 135))

    rows_text = game_font.render(f"Rows: {collector.step}", True, (200, 200, 200))
    screen.blit(rows_text, (info_x, 175))

    # Режим керування
    mode_name = "HUMAN" if control_mode == "human" else "AI"
    mode_color = (100, 220, 100) if control_mode == "ai" else (100, 180, 255)
    mode_text = game_font.render(f"Mode: {mode_name} [M]", True, mode_color)
    screen.blit(mode_text, (info_x, 220))

    # Остання дія AI
    if control_mode == "ai":
        action_text = info_font.render(f"AI action: {last_ai_action}", True, (255, 230, 140))
        screen.blit(action_text, (info_x, 255))

    # Рівень складності (HARD MODE після 250 очок)
    if score >= HARD_MODE_SCORE:
        diff_text = info_font.render("Difficulty: HARD [250+]", True, (255, 90, 90))
        sub_diff_text = help_font.render("Random Blocks Active!", True, (255, 160, 160))
    else:
        diff_text = info_font.render(f"Difficulty: Normal ({score}/{HARD_MODE_SCORE})", True, (130, 220, 130))
        sub_diff_text = help_font.render("Hard mode at 250 pts", True, (140, 170, 140))

    screen.blit(diff_text, (info_x, 285))
    screen.blit(sub_diff_text, (info_x, 308))

    # Віконце наступної фігури
    next_label = info_font.render("Next Piece:", True, (180, 180, 190))
    screen.blit(next_label, (info_x, 335))

    next_box = Rect(info_x, 360, 140, 80)
    draw.rect(screen, (15, 17, 23), next_box)
    draw.rect(screen, (70, 75, 95), next_box, 1)

    next_color = COLORS[next_shape]
    for dx, dy in SHAPES[next_shape][0]:
        nx = info_x + 60 + dx * 20
        ny = 398 + dy * 20
        n_rect = Rect(nx, ny, 18, 18)
        draw.rect(screen, next_color, n_rect)
        draw.rect(screen, (255, 255, 255), n_rect, 1)

    # Підказки керування
    help_title = info_font.render("Controls:", True, (200, 200, 200))
    screen.blit(help_title, (info_x, 455))

    h1 = help_font.render("[M] - Switch Mode (Human/AI)", True, (160, 160, 170))
    h2 = help_font.render("[Arrows] - Move / Rotate", True, (160, 160, 170))
    h3 = help_font.render("[Space] - Hard Drop", True, (160, 160, 170))
    h4 = help_font.render("[R] - Restart Game", True, (160, 160, 170))

    screen.blit(h1, (info_x, 480))
    screen.blit(h2, (info_x, 505))
    screen.blit(h3, (info_x, 530))
    screen.blit(h4, (info_x, 555))

    # Напис Game Over
    if game_over:
        overlay = Surface((BOARD_WIDTH, BOARD_HEIGHT), SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (BOARD_X, BOARD_Y))

        go_text = title_font.render("GAME OVER", True, (255, 60, 60))
        restart_text = info_font.render("Press [R] to Restart", True, (255, 255, 255))
        screen.blit(go_text, (BOARD_X + 35, BOARD_Y + 220))
        screen.blit(restart_text, (BOARD_X + 45, BOARD_Y + 270))

    display.update()

quit()
