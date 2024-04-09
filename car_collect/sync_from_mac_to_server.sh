
#!/bin/bash

rsync -av --delete --exclude-from=.gitignore offroad/ wenli-car@172.26.184.58:~/offroad/offroad
rsync -av --delete --exclude-from=.gitignore data/ wenli-car@172.26.184.58:~/offroad/data
rsync -av --delete --exclude-from=.gitignore video/ wenli-car@172.26.184.58:~/offroad/videols
# rsync -av --delete --exclude-from=.gitignore data cmu@10.42.0.1:/home/cmu/catkin_ws/offroad/data

echo "Sync completed."
