import numpy as np

def input_matrix(name):
    """Takes user input to create a matrix."""
    print(f"\nEnter the dimensions of matrix {name}:")
    rows = int(input("Number of rows: "))
    cols = int(input("Number of columns: "))
    print(f"Enter the elements of matrix {name} row by row (space-separated):")
    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != cols:
            print("Error: Number of elements does not match columns!")
            return None
        matrix.append(row)
    return np.array(matrix)

def display_matrix(matrix, title="Result"):
    """Displays the matrix in a structured format."""
    print(f"\n--- {title} ---")
    print(matrix)
    print("-" * 30)

def main():
    print("===== MATRIX OPERATIONS TOOL =====")
    print("Available operations:")
    print("1. Matrix Addition")
    print("2. Matrix Subtraction")
    print("3. Matrix Multiplication")
    print("4. Matrix Transpose")
    print("5. Determinant Calculation")
    print("6. Exit")

    while True:
        choice = input("\nEnter your choice (1–6): ")

        if choice == '1':
            A = input_matrix('A')
            B = input_matrix('B')
            if A.shape == B.shape:
                display_matrix(A + B, "Matrix A + B")
            else:
                print("Error: Matrices must have the same dimensions for addition.")

        elif choice == '2':
            A = input_matrix('A')
            B = input_matrix('B')
            if A.shape == B.shape:
                display_matrix(A - B, "Matrix A - B")
            else:
                print("Error: Matrices must have the same dimensions for subtraction.")

        elif choice == '3':
            A = input_matrix('A')
            B = input_matrix('B')
            if A.shape[1] == B.shape[0]:
                display_matrix(np.dot(A, B), "Matrix A × B")
            else:
                print("Error: Number of columns of A must equal number of rows of B.")

        elif choice == '4':
            A = input_matrix('A')
            display_matrix(A.T, "Transpose of A")

        elif choice == '5':
            A = input_matrix('A')
            if A.shape[0] == A.shape[1]:
                display_matrix(np.linalg.det(A), "Determinant of A")
            else:
                print("Error: Determinant can only be calculated for square matrices.")

        elif choice == '6':
            print("Exiting Matrix Operations Tool. Goodbye!")
            break

        else:
            print("Invalid choice. Please select from 1–6.")

if __name__ == "__main__":
    main()
