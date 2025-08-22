def nested_data_processing(data):
    # Outer pipeline stage
    intermediate = []
    for item in data:
        # Inner pipeline stage 1
        temp = item * 2

        # Inner pipeline stage 2
        temp += 10

        intermediate.append(temp)

    # Final processing stage
    results = []
    for val in intermediate:
        results.append(val ** 2)

    return results


# print(nested_data_processing([1, 2, 3, 4, 5]))