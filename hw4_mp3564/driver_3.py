#!/usr/bin/env python
#coding:utf-8
# Run program python driver_3.py 
# 20 boards, test using starter text, under minute per board 

import copy, time 
"""
Each sudoku board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8
"""

ROW = "ABCDEFGHI"
COL = "123456789"


def print_board(board):
    """Helper function to print board in a square."""
    print("-----------------")
    for i in ROW:
        row = ''
        for j in COL:
            row += (str(board[i + j]) + " ")
        print(row)

"""Helper function to convert board dictionary to string for writing."""
def board_to_string(board):
    ordered_vals = []
    for r in ROW:
        for c in COL:
            ordered_vals.append(str(board[r + c]))
    return ''.join(ordered_vals)

def count_empty(board):
    count = 0 
    for key in board: 
        if board.get(key) == 0:
            count += 1
    return count   

# returns the keys, row+column of empty cells 
def get_empty_cells(board):
    empty_cells = []
    for key in board:
        if board.get(key) == 0: 
            #empty_cells.append(board[key])
            empty_cells.append(key)
    return empty_cells

"""Helper functions to get rows, columns, and squares"""
def get_row(board, r, key=False):
    row = []
    for c in COL: 
        val = board[r+str(c) + ""]
        if key: 
            variable = r + str(c) + ""
            row.append(variable)
        else: 
            row.append(int(val))
    
    return row 

def get_column(board, c, key=False):
    column = []
    for r in ROW:
        val = board[str(r) + c + ""]
        if key: 
            variable = str(r) + c + ""
            column.append(variable)
        else: 
            column.append(int(val))
    return column 

def get_square(board, r, c, key=False):
    square = []
    row_sets = ['ABC', 'DEF','GHI']
    col_sets = ['123', '456', '789']
    #val = board[str(r+c) + ""]
    for x in row_sets:
        for y in col_sets:
            for i in x :
                for j in y:
                    if r in x and c in y:
                        key_ = i+j 
                        val = board[key_]     
                        if key: 
                            square.append(key_)
                        else: 
                            square.append(int(val))
    return square 


"""Helper functions that check if rows, columns, and squares are valid"""
def check_row(board):
    for r in ROW:
        row = get_row(board, r)
        check_dup = set(row)
        if str(0) in row or len(row) != len(check_dup):
            return False

def check_column(board):
    for c in COL:
        col = get_column(board, c)
        check_dup = set(col)
        if str(0) in col or len(col) != len(check_dup):
            return False

def check_square(board):
    for r in ROW:
        for c in COL:
            sq = get_square(board, r, c)
            check_dup = set(sq)
            if str(0) in sq or len(sq) != len(check_dup):
                return False 
    return True 
# check if row, column, square are empty or have duplicates 
# set gets rid of duplicates, so if len(set) < len(column) there is duplicate
def goal_test(board): 

    if check_column(board) and check_row(board) and check_square(board):
        return True 
    
    return False 

# returns list of possible values to insert into that cell 
def get_valid_values(board, r, c): 
    valid_values = []
    # might have to deal with string issue 
    for i in range(1,10):
        if int(i) not in get_row(board, r) and int(i) not in get_column(board, c) and int(i) not in get_square(board, r, c): #and str(i) not in get_square(board)
            valid_values.append(i)
    return valid_values


# pre-calculates the original legal values/domain of each key 
# returns a dictionary that maps domain to key 
def MRV_empty(board): 
    mrv = []

    empty_cells = get_empty_cells(board)
    
    for i in range(len(empty_cells)):
        row = empty_cells[i][0]
        #print(row)
        col = empty_cells[i][1]
        key = row + col
        # get legal values for each empty space 
        legal = get_valid_values(board, row, col)
        # square = get_square(board, row, col)
        # we want the mrv to have least amount of legal values 
        mrv.append((len(legal), key, legal))
    return sorted(mrv)
    

# return list of tuples of empty cells sorted by minimum restricted values 
# each tuple contains the length of the legal values and the key (row+col)
def MRV(board): 
    min = 10
    mrv_val = "" 
    empty_cells = get_empty_cells(board)
    domain = []
    #empty_val = empty_cells[]
    for empty in empty_cells: 
        # get legal values for each empty space 
        row = empty[0]
        col = empty[1]
        legal = get_valid_values(board, row, col)
        if len(legal) < min:
            min = len(legal)
            mrv_val = empty  
            domain = legal 
    #empty_cells.remove(mrv_val)
    return mrv_val, domain 


def check_cell(board, value, r, c): 
    #print(value)
    if int(value) not in get_row(board, r) and int(value) not in get_column(board, c) and int(value) not in get_square(board, r, c):
        return True 
    return False 

# check that new assignments doesn't cause another variable's domain to go to zero
def forward_check(board):
    empty = get_empty_cells(board)
    for e in empty:
        row = e[0]
        col = e[1]
        if len(get_valid_values(board, row, col)) == 0:
            return False 
    return True 

# given a key, returns a list of values constrained by the same constraints
def get_neighbors(board, row, col):
    # get keys 
    c = get_column(board, col, True)
    r = get_row(board, row, True)
    s = get_square(board, row, col, True)
    neighbors = list(set(c + r + s))
    #print(neighbors)
    return neighbors 

# constraints are get_square, get_row, get_col 
# implements forward_checking from domains of unassigned variables
# removes impossible values 
def inference(board, v, row, col):
    # gets keys of all neighbors 
    neighbors = get_neighbors(board, row, col)
    # remove yourself from the neighbors 
    self_neighbor = row + col
    neighbors.remove(self_neighbor)
    for neighbor in neighbors:
        r = neighbor[0]
        c = neighbor[1]
        # only looking at unassigned 
        if board[neighbor] == 0:
            domain_neighbor = get_valid_values(board, r, c)
            #print(domain_neighbor)
            if v in domain_neighbor:
                # remove value in neighbor's domain
                domain_neighbor.remove(v)
            if len(domain_neighbor) == 0:
                return (False, board)
    return (True, board)
    

def backtrack(board): 
    _, solved_board = backtracking(board)
    return solved_board 
    

# goal test board, choose next variable/found domain using the board
# modify board in for loop, pass in updated board through backtracking recursively 
def backtracking(board):
    """Takes a board and returns solved board."""
    # TODO: implement this
    # A1 would be top left corner, for example  
    # Use minimum remaining value heuristic 
    # Domain = {1, 2, ..., 9}
    # Variables = {A1,..,A9,B1,..,B9}, |V| = 81 
    # 27 constraints 
 
    if len(get_empty_cells(board))==0:#goal_test(board):
        return (True, board)
    
    empty_cells = MRV_empty(board)
    #print(empty_cells[0])
    first_empty = empty_cells[0]
    val = first_empty[1]
    row = val[0]
    col = val[1]
    domain = first_empty[2]
    # gets value, domain using MRV
    #val, domain = MRV(board)
   
    for v in domain:
        # maybe bottle neck this 
        if check_cell(board, v, row, col):
            new_board = copy.deepcopy(board)
            inferences = inference(board, v, row, col)
            #print(inferences[0])
            if inferences[0]:
                board[val] = v
           
                result = backtracking(board)
                if result[0]:
                    return result
            board = new_board
            board[val] = 0 
    return (False, board)     



if __name__ == '__main__':
    #  Read boards from source.
    src_filename = 'sudokus_start.txt'
    try:
        srcfile = open(src_filename, "r")
        sudoku_list = srcfile.read()
    except:
        print("Error reading the sudoku file %s" % src_filename)
        exit()

    # Setup output file
    out_filename = 'output.txt'
    outfile = open(out_filename, "w")

    idx = 0 
    # Solve each board using backtracking
    for line in sudoku_list.split("\n"):

        if len(line) < 9:
            continue
       
        # Parse boards to dict representation, scanning board L to R, Up to Down
        board = { ROW[r] + COL[c]: int(line[9*r+c])
                  for r in range(9) for c in range(9)}
       
        solved_board = backtrack(board)
        
        outfile.write(board_to_string(solved_board))
        
        outfile.write('\n')

    print("Finishing all boards in file.")
