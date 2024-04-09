
#!/bin/bash

# HOST=cmu@172.20.10.2
# HOST=cmu@10.42.0.1
HOST=cmu@rcar.wifi.local.cmu.edu

# rsync -av --delete --exclude-from=.gitignore cmu@10.42.0.1:/home/cmu/catkin_ws/offroad/offroad .
# rsync -av --delete --exclude-from=.gitignore cmu@10.42.0.1:/home/cmu/catkin_ws/offroad/data .
rsync -av $HOST:/home/cmu/catkin_ws/offroad/data/ ./data/
                            
echo "Sync completed."
