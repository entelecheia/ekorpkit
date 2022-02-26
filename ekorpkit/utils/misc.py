def print_status(status):
    max_len_elapsed = max(max(len(row[4]) for row in status), 7)
    max_len_filename = max(max(len(row[5]) for row in status), 9)
    form = "| {:4} | {:25} | {:10} | {:13} | {} | {} |"
    print(
        "\n\n"
        + form.format(
            "Done",
            "Corpus name",
            "Num loaded",
            "Num processed",
            "Elapsed" + " " * (max_len_elapsed - 7),
            "File name" + " " * (max_len_filename - 9),
        )
    )
    print(
        form.format(
            "-" * 4,
            "-" * 25,
            "-" * 10,
            "-" * 13,
            "-" * max_len_elapsed,
            "-" * max_len_filename,
        )
    )
    for finish, name, num_loaded, num_sent, elapsed, filename in status:
        if not filename:
            filename = " " * max_len_filename
        else:
            filename += " " * (max_len_filename - len(filename))
        if not elapsed:
            elapsed = " " * max_len_elapsed
        else:
            elapsed += " " * (max_len_elapsed - len(elapsed))
        print(form.format(finish, name, num_loaded, num_sent, elapsed, filename))
