from langfuse import observe


@observe
def get_page_ranges(page_nums: list[int]) -> list[tuple[int, int] | int]:
    if not page_nums:
        return []
    # Extract unique, sorted integers from the page list
    pages = sorted(list(set(p for p in page_nums if isinstance(p, int))))
    if not pages:
        return []

    ranges = []
    start = pages[0]
    prev = pages[0]

    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            if start == prev:
                ranges.append(start)
            else:
                ranges.append((start, prev))
            start = p
            prev = p

    if start == prev:
        ranges.append(start)
    else:
        ranges.append((start, prev))

    return ranges


@observe
def extract_page_numbers(
    relevant_nodes: list[dict[str, object]], page_count: int, confidence: bool = True
) -> list[int]:
    page_numbers: list[int] = list()
    if not relevant_nodes:
        return page_numbers

    for node in relevant_nodes:
        if not isinstance(node, dict):
            raise TypeError(f"Expected node to be a dictionary, got {type(node).__name__}: {node}")
        
        start_index = node.get("start_index")
        end_index = node.get("end_index")
        
        if start_index is None or end_index is None:
            raise ValueError(f"PageIndex node is missing 'start_index' or 'end_index': {node}")
            
        if not isinstance(start_index, int) or not isinstance(end_index, int):
            raise TypeError(
                f"PageIndex node 'start_index' and 'end_index' must be integers. "
                f"Got start_index={type(start_index).__name__} ({start_index}), "
                f"end_index={type(end_index).__name__} ({end_index})"
            )
            
        if not confidence:
            start_index = (start_index - 1) if start_index > 1 else start_index
            end_index = (end_index + 1) if end_index < page_count else end_index
        page_numbers.append(start_index)
        page_numbers.append(end_index)
    return page_numbers
