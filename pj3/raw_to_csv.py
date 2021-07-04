import csv

# important files
input_file = "new_results.txt"
output_file = "new_results.csv"

# delimiters
run_delimiter = '=' * 80 # between runs of different parameters
gen_delimiter = '=' * 20 # between generations within one run
res_delimiter = '=' * 40 # the final result

def main():
    # read in the results
    with open(input_file, "r") as file:
        results = file.read()
        file.close

    # remove all empty lines
    results_list = results.split('\n')
    while '' in results_list:
        results_list.remove('')

    # recombine the list into a string
    results = '\n'.join(results_list)

    # separate the results by runs
    runs = results.split(run_delimiter)

    # the first set of parameters is the first thing in the list of runs
    params = runs.pop(0)

    # get the output file
    file = open(output_file, 'w')
    writer = csv.writer(file, delimiter=',')

    # for every run
    for run in runs:
        this_p = params.split('\n')
        while '' in this_p:
            this_p.remove('')

        # get the necessary information
        [gens, pop, rates, t_file, _, _, depth] = this_p

        # clean the information
        gens = gens.split(' ')[-1]
        pop = pop.split(' ')[-1]
        rates = ', '.join(''.join(rates.split(' ')).split(':')[-1].split(','))
        t_file = t_file.split(' ')[-1]
        depth = depth.split(' ')[-1]

        try:
            # get the current run data and the next set of parameters
            [data, params] = run.split(res_delimiter)
        except:
            continue

        # get the best final result
        best = (''.join(data.split(res_delimiter))).split('\n')
        while '' in best:
            best.remove('')

        # get just the error from the final result
        final = best[-1].split(' ')[-1]

        # write to a csv
        writer.writerow([gens, pop, rates, t_file, depth, final])

    file.close()

if __name__ == "__main__":
    main()
