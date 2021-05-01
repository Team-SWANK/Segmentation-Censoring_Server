kill $(pgrep -f 'python segment_server.py')
kill $(pgrep -f 'python censor_photos.py')
kill $(pgrep -f 'python detect_segment.py')
python segment_server.py &
sleep 2s
python censor_photos.py &
python detect_segment.py &