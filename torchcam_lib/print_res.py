def process_line(line):
    try:
        # Implement your splitting logic here
        parts = line.split()
        # approssimate to 2 decimal
        val = float(parts[0])
        val = round(val, 2)

        rest = parts[2]

        #splitta in un nome unico la prossima parte del rest da /S: a _imagenette

        S_name = rest.split('S:')[1].split('_')[0]
        T_name = rest.split('T:')[1].split('_')[0]
        kd_name = rest.split('nette_')[1].split('_r')[0]

        xai_name = rest.split('xai:')[1].split('_')[0]
        wxai = rest.split('Wxai:')[1].split('_')[0]

        processed_line = f" {S_name} {T_name} & {kd_name} {xai_name} {wxai} & {val}"

        print(processed_line)
    except Exception as e:
        print(f"Error processing line: {line}", e)
        return ""
    
    return processed_line + '\n'

def copy_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            processed_line = process_line(line.strip())
            f_out.write(processed_line)






if __name__ == "__main__":
    input_file = 'return_of_everything.txt'  # Replace with your input file name
    output_file = 'output1.txt'  # Replace with your output file name
    
    copy_file(input_file, output_file)
