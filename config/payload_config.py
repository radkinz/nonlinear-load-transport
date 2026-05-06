CONFIG = {

    # -------------------------
    # Base / shelf
    # -------------------------
    "base_mass": 8.0,
    "shelf_mass": 3.0,

    # -------------------------
    # Payload geometry
    # -------------------------
    "payload_mass": 20.0,

    "payload_size_x": 0.13,
    "payload_size_y": 0.10,
    "payload_size_z": 0.07,

    "payload_pos_x": 0.12,
    "payload_pos_y": 0.05,
    "payload_pos_z": 0.705,

    # -------------------------
    # Sliding joint dynamics
    # -------------------------
    "slide_range_min": -0.15,
    "slide_range_max": 0.15,

    "payload_damping": 5.0,
    "payload_stiffness": 30.0,

    # -------------------------
    # Contact friction
    # -------------------------
    "friction_slide": 2.0,
    "friction_torsion": 0.1,
    "friction_roll": 0.01,
}