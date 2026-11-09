# show the intuition of how a general matrix can be formated in CSR (Compressed Sparse Row)

if __name__=="__main__":
    A = [
        [10, 0, 0, 2], 
        [0 , 3, 0, 0], 
        [0 , 0, 0, 0], 
        [4 , 0, 5, 6]
    ]

    m, n = 4, 4
    print(f"matrix: {A}\n")

    csr = {
        "val": [],
        "col": [],
        "row_ptr": [0]
    }
    for i in range(m):
        per_row=0
        for j in range(n):
            aij = A[i][j]
            if aij != 0: # every no-zero add its value and column to csr, the 'row_ptr' is updated by row
                per_row += 1
                csr["val"].append(aij)
                csr["col"].append(j)
        # 'row_ptr' give you the number of no-zero elements in a row
        # row=i non-zero = row_ptr[i+1] - row_ptr[i], since row_ptr[i] count the non-zero elements until this row.
        csr["row_ptr"].append(csr["row_ptr"][-1] + per_row)
        print(f"row={i}, csr={csr}")

    print(f"\ncsr={csr}")
    d = csr['row_ptr'][-1]/(m*n)
    print(f"density(A)={d}")

            
