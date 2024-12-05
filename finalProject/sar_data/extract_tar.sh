# Run this in the directory where downloaded tar.gz files are located
for file in *.tar.gz; do
    if [ -f "$file" ]; then
        tar xzvf "$file"
        rm "$file"
    fi
done