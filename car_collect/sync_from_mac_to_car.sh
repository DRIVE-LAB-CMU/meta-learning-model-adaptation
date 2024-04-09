
#!/bin/bash

# HOST=cmu@172.20.10.2
# HOST=cmu@10.42.0.1
HOST=cmu@rcar.wifi.local.cmu.edu

rsync -av --delete --exclude-from=.gitignore offroad/ $HOST:/home/cmu/catkin_ws/offroad/offroad
# rsync -av --delete --exclude-from=.gitignore tmp/ $HOST:/home/cmu/catkin_ws/offroad/tmp
# rsync -av --delete --exclude-from=.gitignore offroad/ cmu@172.26.64.101:/home/cmu/catkin_ws/offroad/offroad
# rsync -av --delete --exclude-from=.gitignore data cmu@10.42.0.1:/home/cmu/catkin_ws/offroad/data

echo "Sync completed."
