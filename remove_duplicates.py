def remove_duplicate_lines(input_file_path, output_file_path):
    """
    Removes duplicate lines from a text file in a memory-efficient way
    and writes the unique lines to an output file.

    This script reads the input file line by line, which means it does not
    load the entire file into memory. It uses a set to keep track of the
    lines it has already seen, which is very fast for checking duplicates.

    NOTE: While the file itself is not loaded into memory, this script does
    store all unique lines in a set in memory. If the number of *unique* lines
    is exceptionally large (e.g., billions), this could still consume a
    significant amount of RAM.
    """
    seen_lines = set()
    try:
        with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            
            print(f"Processing '{input_file_path}'...")
            line_count = 0
            duplicate_count = 0

            for line in infile:
                line_count += 1
                if line not in seen_lines:
                    outfile.write(line)
                    seen_lines.add(line)
                else:
                    duplicate_count += 1
                
                if line_count % 1000000 == 0:
                    print(f"  ...processed {line_count:,} lines")

            print("\nProcessing complete.")
            print(f"Total lines read: {line_count:,}")
            print(f"Duplicate lines found and removed: {duplicate_count:,}")
            print(f"Unique lines written to '{output_file_path}': {len(seen_lines):,}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")