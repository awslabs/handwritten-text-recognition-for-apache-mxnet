#!/bin/sh


dos2unix rover.htm | \
    sed -e 's/SIZE=5>/SIZE=8>/g' \
	-e 's/SIZE=4>/SIZE=7>/g' \
	-e 's/SIZE=3>/SIZE=6>/g' \
	-e 's/SIZE=2>/SIZE=5>/g' \
	-e 's/SIZE=1>/SIZE=4>/g' \

