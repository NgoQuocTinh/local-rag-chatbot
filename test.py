def removeDuplicates(nums):
    print("Initial nums:", nums)
    print("-" * 40)

    if not nums:
        return 0

    k = 1
    print(f"Start: k = {k} (nums[0] = {nums[0]} is always unique)")
    print("-" * 40)

    for i in range(1, len(nums)):
        print(f"Step i = {i}")
        print(f"Compare nums[{i}] = {nums[i]} with nums[{i-1}] = {nums[i-1]}")

        if nums[i] != nums[i - 1]:
            nums[k] = nums[i]
            print(f"  -> Different! Write nums[{k}] = {nums[i]}")
            k += 1
        else:
            print("  -> Same! Skip")

        print("Current nums:", nums)
        print(f"Current k = {k}")
        print("-" * 40)

    print("Finished!")
    print(f"Result k = {k}")
    print("Unique part of nums:", nums[:k])
    return k


nums = [1, 1, 2, 2, 3]
removeDuplicates(nums)
