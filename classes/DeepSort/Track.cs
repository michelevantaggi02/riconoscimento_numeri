using System.Numerics;
using OpenCvSharp;

namespace riconoscimento_numeri.classes.DeepSort
{
    public class Track
    {
        public int id { get; set; }

        public List<Rect> history { get; set; }

        public Rect currentBounds { get; set; }

        public Rect predictedNextBounds { get; set; }

        public Detail medianAppearance { get; private set; }

        private List<Detail> appearances { get; set; }

        public int missedFrames { get; set; }
        public int consecutiveHits { get; set; }
        public int lifeTime { get; set; }

        public int trackLimit { get; init; }



        public Track(int id, Rect bounds, Detail appearances, int trackLimit)
        {
            history = [];

            this.id = id;
            currentBounds = bounds;
            history = [bounds];
            medianAppearance = appearances;
            this.appearances = [appearances];
            this.trackLimit = trackLimit;
        }

        public void Register(Rect bounds, Detail appearance)
        {
            currentBounds = bounds;
            history.Add(bounds);
            appearances.Add(appearance);

            UpdateMedianAppearance();
        }

        public void Update()
        {
            missedFrames = 0;
            consecutiveHits++;
            lifeTime++;

        }

        private void UpdateMedianAppearance()
        {

            medianAppearance = appearances[0];
            for(int i = 1; i < appearances.Count; i++)
            {
                medianAppearance += appearances[i];
            }

            medianAppearance.Normalize();
        }


        public void Draw(Mat image)
        {

            foreach (Rect rect in history)
            {
                Cv2.DrawMarker(image, rect.BottomRight, Scalar.Green, MarkerTypes.Cross, 10);
            }

            Cv2.Rectangle(image, currentBounds, Scalar.Red, 2);
            Cv2.PutText(image, $"ID: {id}", currentBounds.BottomRight, HersheyFonts.HersheySimplex, 1, Scalar.Black, 6);
            Cv2.PutText(image, $"ID: {id}", currentBounds.BottomRight, HersheyFonts.HersheySimplex, 1, Scalar.White, 2);


        }


        //TODO: CHECK IMPLEMENTATION https://github.com/KQTENQK/MOT-DeepSort-CS/blob/main/src/MOT.CORE/Matchers/Trackers/KalmanTracker.cs
        public Rect PredictNextBounds()
        {
            if(missedFrames > 0)
            {
                consecutiveHits = 0;
            }

            missedFrames++;

            if (history.Count < 2)
            {
                Console.WriteLine("Not enough history, giving current");
                Console.WriteLine($"{id} Predicted: ({currentBounds.Left},{currentBounds.Top})({currentBounds.Right},{currentBounds.Bottom})");

                predictedNextBounds = currentBounds;
                return currentBounds;
            }

            Rect last = history[^1];
            Rect beforeLast = history[^2];
            int widthChange = last.Width - beforeLast.Width;
            int heightChange = last.Height - beforeLast.Height;

            int velocityX = last.Left - beforeLast.Left;
            int velocityY = last.Top - beforeLast.Top;



            predictedNextBounds = new Rect(last.Left + velocityX, last.Top + velocityY, last.Width + widthChange, last.Height + heightChange);

            Console.WriteLine($"{id} Predicted: ({predictedNextBounds.Left},{predictedNextBounds.Top})({predictedNextBounds.Right},{predictedNextBounds.Bottom})");
            return predictedNextBounds;
        }

    }
}
