import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, FancyArrowPatch, Rectangle

# ============================================================
# OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# COMMON HELPERS
# ============================================================

COLORS = {
    "bg": "#F8FAFC",
    "dark": "#0F172A",
    "text": "#1E293B",
    "muted": "#64748B",
    "pink": "#EC4899",
    "pink_light": "#FCE7F3",
    "purple": "#7C3AED",
    "purple_light": "#EDE9FE",
    "blue": "#2563EB",
    "blue_light": "#DBEAFE",
    "green": "#059669",
    "green_light": "#D1FAE5",
    "orange": "#F59E0B",
    "orange_light": "#FEF3C7",
    "border": "#334155",
    "line": "#475569"
}

def new_canvas(title, filename, width=14, height=10):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")

    ax.text(
        7, 9.55, title,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=COLORS["dark"]
    )
    return fig, ax

def save_canvas(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created: {path}")

def draw_box(ax, x, y, w, h, title, lines=None, fill="#FFFFFF", border=None):
    if border is None:
        border = COLORS["border"]

    shadow = FancyBboxPatch(
        (x + 0.05, y - 0.05), w, h,
        boxstyle="round,pad=0.04",
        linewidth=0,
        facecolor="#CBD5E1",
        alpha=0.35
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04",
        linewidth=2,
        edgecolor=border,
        facecolor=fill
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2, y + h - 0.35,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["dark"]
    )

    if lines:
        current_y = y + h - 0.8
        for line in lines:
            ax.text(
                x + 0.22,
                current_y,
                line,
                ha="left",
                va="center",
                fontsize=9,
                color=COLORS["text"]
            )
            current_y -= 0.34

def draw_ellipse(ax, x, y, w, h, text, fill="#FFFFFF", border=None):
    if border is None:
        border = COLORS["pink"]

    ellipse = Ellipse(
        (x, y),
        width=w,
        height=h,
        facecolor=fill,
        edgecolor=border,
        linewidth=2
    )
    ax.add_patch(ellipse)

    ax.text(
        x, y,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=COLORS["text"]
    )

def draw_arrow(ax, start, end, label=None, dashed=False):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
        linestyle="--" if dashed else "-",
        color=COLORS["line"]
    )
    ax.add_patch(arrow)

    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(
            mid_x,
            mid_y + 0.15,
            label,
            ha="center",
            fontsize=8.5,
            color=COLORS["muted"]
        )

def draw_actor(ax, x, y, name, color=None):
    if color is None:
        color = COLORS["pink"]

    head = Circle((x, y + 0.45), 0.18, edgecolor=color, facecolor="white", linewidth=2)
    ax.add_patch(head)

    ax.plot([x, x], [y + 0.27, y - 0.35], color=color, linewidth=2)
    ax.plot([x - 0.35, x + 0.35], [y + 0.05, y + 0.05], color=color, linewidth=2)
    ax.plot([x, x - 0.25], [y - 0.35, y - 0.8], color=color, linewidth=2)
    ax.plot([x, x + 0.25], [y - 0.35, y - 0.8], color=color, linewidth=2)

    ax.text(
        x, y - 1.15,
        name,
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["dark"]
    )

# ============================================================
# 1. USE CASE DIAGRAM
# ============================================================

def create_use_case_diagram():
    fig, ax = new_canvas("T&A IntelliMart - Use Case Diagram", "Use_Case_Diagram.png")

    # System boundary
    system = FancyBboxPatch(
        (3.1, 0.7), 9.5, 8.2,
        boxstyle="round,pad=0.08",
        facecolor="#FFFFFF",
        edgecolor=COLORS["border"],
        linewidth=2.3
    )
    ax.add_patch(system)

    ax.text(
        7.85, 8.55,
        "T&A IntelliMart System",
        ha="center",
        fontsize=13.5,
        fontweight="bold",
        color=COLORS["dark"]
    )

    # Actors
    draw_actor(ax, 1.3, 7.2, "Customer", COLORS["blue"])
    draw_actor(ax, 1.3, 4.2, "Admin", COLORS["pink"])
    draw_actor(ax, 1.3, 1.9, "System", COLORS["green"])

    # Customer use cases
    draw_ellipse(ax, 6, 7.6, 3.0, 0.8, "Login")
    draw_ellipse(ax, 6, 6.55, 3.0, 0.8, "Browse Categories")
    draw_ellipse(ax, 6, 5.5, 3.0, 0.8, "View Products")
    draw_ellipse(ax, 9.8, 6.55, 3.1, 0.8, "View Price Analysis")
    draw_ellipse(ax, 9.8, 5.5, 3.1, 0.8, "Add / View Reviews")

    # Admin use cases
    draw_ellipse(ax, 6, 3.9, 3.0, 0.8, "View Analytics")
    draw_ellipse(ax, 9.8, 3.9, 3.1, 0.8, "Manage Reviews")

    # System use cases
    draw_ellipse(ax, 6, 2.0, 3.0, 0.8, "Scrape Product Data", fill=COLORS["green_light"], border=COLORS["green"])
    draw_ellipse(ax, 9.8, 2.0, 3.1, 0.8, "Run ML Predictions", fill=COLORS["purple_light"], border=COLORS["purple"])

    # Connections
    customer_point = (1.65, 7.2)
    for target in [(4.5, 7.6), (4.5, 6.55), (4.5, 5.5), (8.25, 6.55), (8.25, 5.5)]:
        ax.plot([customer_point[0], target[0]], [customer_point[1], target[1]], color=COLORS["line"], linewidth=1.4)

    admin_point = (1.65, 4.2)
    for target in [(4.5, 3.9), (8.25, 3.9)]:
        ax.plot([admin_point[0], target[0]], [admin_point[1], target[1]], color=COLORS["line"], linewidth=1.4)

    system_point = (1.65, 1.9)
    for target in [(4.5, 2.0), (8.25, 2.0)]:
        ax.plot([system_point[0], target[0]], [system_point[1], target[1]], color=COLORS["line"], linewidth=1.4)

    save_canvas(fig, "Use_Case_Diagram.png")

# ============================================================
# 2. ER DIAGRAM
# ============================================================

def create_er_diagram():
    fig, ax = new_canvas("T&A IntelliMart - Entity Relationship Diagram", "ER_Diagram.png")

    def draw_table(x, y, title, rows, color):
        draw_box(ax, x, y, 3.4, 2.5, title, rows, fill="#FFFFFF", border=color)
        # Header fill
        ax.add_patch(Rectangle((x, y + 2.08), 3.4, 0.42, facecolor=color, edgecolor=color))
        ax.text(x + 1.7, y + 2.29, title, ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    draw_table(0.8, 6.3, "User", [
        "user_id (PK)",
        "username",
        "role"
    ], COLORS["blue"])

    draw_table(5.3, 6.3, "Product", [
        "product_id (PK)",
        "product_name",
        "category",
        "final_price",
        "vendor"
    ], COLORS["orange"])

    draw_table(9.8, 6.3, "Review", [
        "review_id (PK)",
        "product_name (FK)",
        "username",
        "rating",
        "review_text"
    ], COLORS["green"])

    draw_table(5.3, 2.5, "Price Analysis", [
        "analysis_id (PK)",
        "product_name (FK)",
        "predicted_price",
        "discount_suggestion"
    ], COLORS["purple"])

    draw_table(9.8, 2.5, "Vendor", [
        "vendor_id (PK)",
        "vendor_name",
        "rating"
    ], COLORS["pink"])

    # Relationships
    draw_arrow(ax, (4.2, 7.5), (5.3, 7.5), "browses")
    draw_arrow(ax, (8.7, 7.5), (9.8, 7.5), "has reviews")
    draw_arrow(ax, (7.0, 6.3), (7.0, 5.0), "used in")
    draw_arrow(ax, (8.7, 7.0), (9.8, 4.0), "sold by")

    save_canvas(fig, "ER_Diagram.png")

# ============================================================
# 3. DATA FLOW DIAGRAM
# ============================================================

def create_data_flow_diagram():
    fig, ax = new_canvas("T&A IntelliMart - Data Flow Diagram", "Data_Flow_Diagram.png")

    # External entities
    draw_box(ax, 0.7, 7.2, 2.3, 1.0, "Customer", ["Input / View"], fill=COLORS["blue_light"], border=COLORS["blue"])
    draw_box(ax, 0.7, 4.7, 2.3, 1.0, "Admin", ["Manage / Analyze"], fill=COLORS["pink_light"], border=COLORS["pink"])
    draw_box(ax, 0.7, 2.2, 2.3, 1.0, "External Sites", ["Amazon / Flipkart"], fill=COLORS["green_light"], border=COLORS["green"])

    # Processes
    draw_box(ax, 4.2, 7.0, 3.2, 1.2, "1. Authentication", ["Login / role check"], fill="#FFFFFF", border=COLORS["border"])
    draw_box(ax, 4.2, 5.0, 3.2, 1.2, "2. Product Search", ["Category / product query"], fill="#FFFFFF", border=COLORS["border"])
    draw_box(ax, 4.2, 3.0, 3.2, 1.2, "3. Review System", ["Add / fetch reviews"], fill="#FFFFFF", border=COLORS["border"])
    draw_box(ax, 8.8, 5.0, 3.2, 1.2, "4. ML Analysis", ["Price prediction"], fill="#FFFFFF", border=COLORS["border"])
    draw_box(ax, 8.8, 3.0, 3.2, 1.2, "5. Data Scraping", ["Collect product data"], fill="#FFFFFF", border=COLORS["border"])

    # Data stores
    draw_box(ax, 4.2, 1.0, 3.2, 1.0, "D1 SQLite DB", ["Products + Reviews"], fill=COLORS["orange_light"], border=COLORS["orange"])
    draw_box(ax, 8.8, 1.0, 3.2, 1.0, "D2 ML Model", ["Random Forest"], fill=COLORS["purple_light"], border=COLORS["purple"])

    # Arrows
    draw_arrow(ax, (3.0, 7.7), (4.2, 7.7), "login data")
    draw_arrow(ax, (3.0, 5.2), (4.2, 5.6), "query")
    draw_arrow(ax, (3.0, 5.0), (4.2, 3.6), "review")
    draw_arrow(ax, (3.0, 2.7), (8.8, 3.6), "scraped data")

    draw_arrow(ax, (7.4, 5.6), (8.8, 5.6), "features")
    draw_arrow(ax, (10.4, 5.0), (10.4, 2.0), "model data")
    draw_arrow(ax, (5.8, 3.0), (5.8, 2.0), "store")
    draw_arrow(ax, (5.8, 5.0), (5.8, 2.0), "read/write")

    save_canvas(fig, "Data_Flow_Diagram.png")

# ============================================================
# 4. SEQUENCE DIAGRAM
# ============================================================

def create_sequence_diagram():
    fig, ax = new_canvas("T&A IntelliMart - Price Prediction Sequence Diagram", "Sequence_Diagram.png")

    participants = [
        ("Customer", 1.5, COLORS["blue"]),
        ("Frontend", 4.5, COLORS["pink"]),
        ("Flask API", 7.5, COLORS["orange"]),
        ("ML Model", 10.5, COLORS["purple"]),
        ("SQLite DB", 13.0, COLORS["green"]),
    ]

    # Draw participant headers and lifelines
    for name, x, color in participants:
        draw_box(ax, x - 0.8, 8.6, 1.6, 0.6, name, [], fill="#FFFFFF", border=color)
        ax.plot([x, x], [8.6, 1.0], color="#CBD5E1", linewidth=2, linestyle="--")

    def msg(y, x1, x2, text):
        draw_arrow(ax, (x1, y), (x2, y), text)

    msg(7.8, 1.5, 4.5, "select category")
    msg(7.0, 4.5, 7.5, "GET /api/products")
    msg(6.2, 7.5, 13.0, "fetch product data")
    msg(5.4, 13.0, 7.5, "return products")
    msg(4.6, 7.5, 10.5, "send features")
    msg(3.8, 10.5, 7.5, "predicted price")
    msg(3.0, 7.5, 4.5, "JSON response")
    msg(2.2, 4.5, 1.5, "show results")

    save_canvas(fig, "Sequence_Diagram.png")

# ============================================================
# RUN ALL DIAGRAMS
# ============================================================

create_use_case_diagram()
create_er_diagram()
create_data_flow_diagram()
create_sequence_diagram()

print("\n🎉 All diagrams generated successfully!")
print("Files created:")
print("1. Use_Case_Diagram.png")
print("2. ER_Diagram.png")
print("3. Data_Flow_Diagram.png")
print("4. Sequence_Diagram.png")