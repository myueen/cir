import csv
from CIR import CIR

# with open("foregroundX.csv", 'r') as file:
#     csvreader = csv.reader(file)

#     for row in csvreader:
#         print(row)

# Test code
X = [[1, 4, 5],
     [-5, 8, 9],
     [2, 3, 6]]

X1 = [[10, 18, 5],
      [-8, 11, 12],
      [2, 17, 2]]

X2 = [[55, 28, 34],
      [30, 32, 35],
      [45, 29, 28]]

X3 = [[120, 28, 34],
      [30, 98, 35],
      [45, 103, 28],
      [33, 62, 59],
      [89, 75, 28]]

X4 = [[120, 23, 12],
      [34, 12, 34], 
      [87, 73, 92], 
      [73, 69, 57],
      [65, 88, 72], 
      [104, 112, 38],
      [123, 77, 38]]

Y_numerical = [1, 8.3, 2.7]
Y1 = [1, 13, 20, 55, 23]
Y_binary = [0, 1, 1, 0, 1, 1, 0, 1, 0]
Y_unique_less_10 = [20, 40, 23, 10, 39, 72, 42]
Y_categorical = ["a", "b", "b", "c", "d", "d", "e"]

Xt = [[1, 4, 5],
      [-5, 8, 9],
      [28, 21, 14],
      [1, 15, 5],
      [18, 8, 20],
      [28, 34, 14]]

Yt = [1, 9, 21, 30, 20, 40]


result = CIR(X4, Y_unique_less_10, Xt, Yt, 2, 3)
print(result)
# print(result[0])
# print(result[1])


# ====================================================================================
    # The following are print statement for code testing.
    # print("Original Matix")
    # print(X_df)
    # print("             ")
    # print("Centered Matix")
    # print(X_centered)
    # print("             ")
    # print("Covariance Matix for X")
    # print(X_cov_matrix)
    # print("             ")
    # print("Y original 1d array")
    # print(Y_df)
    # print("             ")
    # print("Number of unique value in Y")
    # print(Y_unique_value)
    # print("             ")
    # print("H: # intervals split range(Y)")
    # print(H)
    # print("             ")
    # print("Intervals and count: ")
    # print(Ph)