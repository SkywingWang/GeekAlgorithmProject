package algorithm;

/**
 * 304. 二维区域和检索 - 矩阵不可变
 *
 * 给定一个二维矩阵，计算其子矩形范围内元素的总和，该子矩阵的左上角为 (row1, col1) ，右下角为 (row2, col2) 。
 *
 *
 * 上图子矩阵左上角 (row1, col1) = (2, 1) ，右下角(row2, col2) = (4, 3)，该子矩形内元素的总和为 8。
 *
 *
 */
public class NumMatrix {
    int[][] sums;
    public NumMatrix(int[][] matrix) {
        int m = matrix.length;
        if(m > 0){
            int n = matrix[0].length;
            sums = new int[m][n+1];
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    sums[i][j+1] = sums[i][j] + matrix[i][j];
                }
            }
        }
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        int sum = 0;
        for(int i = row1; i <= row2;i++){
            sum += sums[i][col2 + 1] - sums[i][col1];
        }
        return sum;
    }
}
