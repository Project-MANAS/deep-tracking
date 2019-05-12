# Deep tracking ROS wrapper

## Nodes required for the deep tracking pipeline:
- deep-tracking/deep_tracker.py
- ray_caster/ray_caster
- map_to_jpeg/map_to_image_node
- objecttracker/tracker_node.py

## Data flow and associated topics:
- Costmap obstacle layer from ```/move_base/local_costmap/obstacle_layer/obstacle_map``` -> deep_tracker.py
- Local Costmap from ```/move_base/local_costmap/costmap``` -> map_to_image_node -> ```/map_image/full```
- ```/map_image/full``` -> ray_caster -> ```visibility_layer```
- ```visibility_layer``` -> deep_tracker.py
