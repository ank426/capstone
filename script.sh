#!/bin/sh

if [ $# -eq 0 ]; then
  echo "Usage: $0 file1 [file2 file3 ...]"
  echo "Please provide at least one file as an argument."
  exit 1
fi

random_choice() {
  echo $(( $(od -An -N1 -i /dev/urandom) % 4 ))
}

for file in "$@"; do
  if [ ! -f "$file" ] || [ ! -r "$file" ]; then
    echo "Error: Cannot read file '$file'"
    continue
  fi

  line_number=0
  while IFS= read -r line; do
    line_number=$((line_number + 1))

    choice=$(random_choice)

    case $choice in
      1)
        echo "$file:$line_number: error: Error"
        ;;
      2)
        echo "$file:$line_number: warning: Warning"
        ;;
      3)
        echo "$file:$line_number: note: Note"
        ;;
      0|*)
        ;;
    esac
  done < "$file"
done

exit 0
