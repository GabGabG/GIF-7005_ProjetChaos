from manim import *
from src.EDOs import Lorenz

class LorenzAttractor(ThreeDScene):
    def construct(self):
        # Initialiser le système de Lorenz
        lorenz_system = Lorenz((1,0,0))

        # Résoudre l'EDO - simulation complète
        lorenz_system.resoudre_EDO(t_min=0.0, t_max=40.0, nombre_t=8000)

        # Extraire les points pour la trajectoire
        x_points = lorenz_system.x_points[0]
        y_points = lorenz_system.x_points[1]
        z_points = lorenz_system.x_points[2]

        # Créer la trajectoire 3D
        def lorenz_func(t):
            index = int(t * (len(x_points) - 1) / 40)  # Ajusté pour t_max=40
            if index >= len(x_points):
                index = len(x_points) - 1
            return np.array([
                x_points[index] * 0.1,
                y_points[index] * 0.1,
                z_points[index] * 0.1
            ])
        
        trajectory = ParametricFunction(
            lorenz_func,
            t_range=[0, 40, 0.05],  # Full simulation time range
            color=BLUE,
            stroke_width=4
        )

        # Axes 3D with longer z-axis
        axes = ThreeDAxes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            z_range=[-2, 7, 2],  # Extended z-axis
            z_length=7  # Make z-axis physically longer
        )
        
        # Set camera angle
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        
        # Shift system down by about 1/4 of picture height (use Z direction for proper vertical shift)
        axes.shift([0, 0, -1.5])
        trajectory.shift([0, 0, -1.5])
        
        self.add(axes)
        # Full length animation (26 seconds total):
        self.play(Create(trajectory), run_time=20)  # Extended creation time
        
        # Add camera rotation to show the 3D structure
        self.begin_ambient_camera_rotation(rate=0.3)  # Slower for longer duration
        self.wait(4)  # Extended rotation time
        self.stop_ambient_camera_rotation()
        self.wait(2)  # Final pause