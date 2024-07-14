# pcl_proc

Package to filter msis_pcl pointclouds

### How to run
- `roslaunch rise_multirobot bringup_simulation.launch` (or any other vehicle, but change the configs here too accordingly)
- `rosservice call /<vehicle_name>/controllers/enable`
- `roslaunch pcl_proc wp_admin.launch` to start the guidance around the iceberg.

[Video](https://drive.google.com/file/d/1_OjvJ9xO-Ar6HdSANdNQEcmdRiWNHEWJ/view?usp=drive_link)