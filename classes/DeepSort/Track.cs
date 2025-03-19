using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
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
            Vector2 last = new(history[^1].X, history[^1].Y);
            Vector2 beforeLast = new(history[^2].X, history[^2].Y);
            Vector2 velocity = last - beforeLast;
            Vector2 next = last + velocity;
            predictedNextBounds = new Rect((int)next.X, (int)next.Y, currentBounds.Width, currentBounds.Height);

            Console.WriteLine($"{id} Predicted: ({predictedNextBounds.Left},{predictedNextBounds.Top})({predictedNextBounds.Right},{predictedNextBounds.Bottom})");
            return predictedNextBounds;
        }

    }
}
