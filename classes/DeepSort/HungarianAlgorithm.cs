

using System.Numerics;

namespace riconoscimento_numeri.classes.DeepSort
{


    public class HungarianAlgorithm
    {
        private enum State : byte
        {
            First,
            Second,
            Third,
            Fourth,
            Final
        }

        private enum MaskValue : byte
        {
            Undefined = 0,
            Star,
            Prime
        }



        private double[,] costs;

        private int height;
        private int width;
        private bool[] crossedRows;
        private bool[] crossedCols;
        private bool resized = false;

        private MaskValue[,] mask;

        private (int y, int x) pathStart;
        private (int y, int x)[] path;


        public HungarianAlgorithm(double[,] costMatrix)
        {
            costs = InitCosts(costMatrix);
            height = costs.GetLength(0);
            width = costs.GetLength(1);
            mask = new MaskValue[height, width];
            crossedRows = new bool[height];
            crossedCols = new bool[width];

            path = new (int y, int x)[height * width];
            pathStart = (0, 0);
        }

        private double[,] InitCosts(double[,] costMatrix)
        {
            int heigth = costMatrix.GetLength(0);
            int width = costMatrix.GetLength(1);

            if (heigth > width)
            {
                double[,] newCosts = new double[width, heigth];
                for (int i = 0; i < heigth; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        newCosts[j, i] = costMatrix[i, j];
                    }
                }

                resized = true;

                return newCosts;
            }
            else
            {
                return costMatrix;
            }
        }

        public int[] Solve()
        {
            InitMask();

            UnCross();

            State state = State.First;

            while (state != State.Final)
            {
                state = Execute(state);
            }

            return Assign();
        }


        private State Execute(State current)
        {
            return current switch
            {
                State.First => FirstState(),
                State.Second => SecondState(),
                State.Third => ThirdState(),
                State.Fourth => FourthState(),
                _ => State.Final,
            };
        }

        private State FirstState()
        {
            Cross();

            if (CountCrossedCols() == height)
                return State.Final;

            return State.Second;
        }

        private State SecondState()
        {
            while (true)
            {
                (int y, int x)? zero = FindZero();

                if (zero == null)
                    return State.Fourth;

                mask[zero.Value.y, zero.Value.x] = MaskValue.Prime;

                int col = FindMaskCol(zero.Value.y, MaskValue.Star);

                if (col != -1)
                {
                    crossedRows[zero.Value.y] = true;
                    crossedCols[col] = false;
                }
                else
                {
                    pathStart = zero.Value;
                    return State.Third;
                }
            }
        }

        private State ThirdState()
        {
            int pathIdx = 0;
            path[pathIdx] = pathStart;

            while (true)
            {
                int row = FindMaskRow(path[pathIdx].x, MaskValue.Star);
                if (row == -1)
                    break;
                pathIdx++;
                path[pathIdx] = (row, path[pathIdx - 1].x);
                int col = FindMaskCol(path[pathIdx].y, MaskValue.Prime);
                pathIdx++;
                path[pathIdx] = (path[pathIdx - 1].y, col);
            }

            ReduceMask(pathIdx + 1);
            UnCross();
            UndefinePrimes();
            return State.First;

        }

        private State FourthState()
        {
            double min = MinCost();

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                {
                    if (crossedRows[i])
                        costs[i, j] += min;
                    if (!crossedCols[j])
                        costs[i, j] -= min;
                }

            return State.Second;
        }

        private int[] Assign()
        {
            int size = height;

            if (resized)
            {
                size = width;
            }
            int[] assigned = new int[size];
            Array.Fill(assigned, -1);

            for(int i = 0; i < height; i++)
                for(int j = 0; j < width; j++)
                    if (mask[i, j] == MaskValue.Star)
                    {
                        if (resized)
                            assigned[j] = i;
                        else
                            assigned[i] = j;

                        break;

                    }

            return assigned;
        }

        private void InitMask()
        {
            for (int i = 0; i < height; i++)
            {
                double min = double.MaxValue;

                for (int j = 0; j < width; j++)
                    min = double.Min(min, costs[i, j]);

                for (int j = 0; j < width; j++)
                {
                    costs[i, j] -= min;

                    if (costs[i, j] == 0 && !crossedRows[i] && !crossedCols[j])
                    {
                        mask[i, j] = MaskValue.Star;
                        crossedRows[i] = true;
                        crossedCols[j] = true;
                    }
                }
            }
        }

        private void UnCross()
        {

            int i = 0;
            int j = 0;

            int diff = int.Abs(height - width);

            if (height > width)
            {
                for (; i < diff; i++)
                    crossedRows[i] = false;
            }
            else
            {
                for (; j < diff; j++)
                    crossedCols[j] = false;
            }


            for (; i < height; i++)
                crossedRows[i] = false;

            for (; j < width; j++)
                crossedCols[j] = false;

        }

        private (int, int)? FindZero()
        {
            for(int i = 0; i < height; i++)
                for(int j = 0; j < width; j++)
                    if(costs[i, j] == 0.0 && !crossedCols[j] && !crossedRows[i])
                        return (i, j);

            return null;
        }

        private int FindMaskCol(int row, MaskValue value)
        {
            for (int j = 0; j < width; j++)
                if (mask[row, j] == value)
                    return j;
            return -1;
        }

        private int FindMaskRow(int col, MaskValue value)
        {
            for (int i = 0; i < height; i++)
                if (mask[i, col] == value)
                    return i;
            return -1;
        }

        private int CountCrossedCols()
        {
            return crossedCols.Where((col) => col).Count(); 
        }

        private void Cross()
        {
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    if (mask[i, j] == MaskValue.Star)
                        crossedCols[j] = true;
        }


        private void ReduceMask(int pathLen)
        {
            for (int i = 0; i < pathLen; i++)
            {
                mask[path[i].y, path[i].x] = mask[path[i].y, path[i].x] switch
                {
                    MaskValue.Star => MaskValue.Undefined,
                    MaskValue.Prime => MaskValue.Star,
                    _ => MaskValue.Undefined
                };
            }
        }

        private void UndefinePrimes()
        {
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    if (mask[i, j] == MaskValue.Prime)
                        mask[i, j] = MaskValue.Undefined;
        }

        private double MinCost()
        {
            double min = double.MaxValue;
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    if (!crossedRows[i] && !crossedCols[j])
                        min = double.Min(min, costs[i, j]);
            return min;
        }
    }
}
