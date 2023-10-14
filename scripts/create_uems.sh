#!/bin/sh

convert_to_seconds() {
  local time_str="$1"
  local minutes=$(echo "$time_str" | cut -d: -f1)
  local seconds=$(echo "$time_str" | cut -d: -f2 | cut -d. -f1)
  local milliseconds=$(echo "$time_str" | cut -d. -f2)
  
  # Convert to seconds (keeping 2 decimal places for milliseconds)
  echo "scale=3; ($minutes * 60) + $seconds + ($milliseconds / 100)" | bc
}

AUDIO_LIST="../lists/audio-list.txt"

while IFS= read -r line; do
  # Check if the line contains URI and time pattern
  match=$(echo "$line" | grep '[a-zA-Z0-9]\+ [0-9]\+:[0-9]\+\.[0-9]\+')
  if [ ! -z "$match" ]; then
    # Extract uri and time
    uri=$(echo "$line" | awk '{print $2}')
    time=$(echo "$line" | awk '{print $3}')
    start="0.000"
    end=$(convert_to_seconds "$time")

    uem="../uems/${uri}.uem"

    echo "${uri} 1 ${start} ${end}" > $uem
  fi

done < "$AUDIO_LIST"
