using OpenCvSharp;
using YoloDotNet.Models;

namespace riconoscimento_numeri.classes.DeepSort
{
    /// <summary>
    /// DeepSort implementation.
    /// </summary>
    public class Matcher
    {
        private List<Track> tracks;
        private YoloRecognizer yoloPredictor;
        private FastReID reidAppExtractor;

        private int trackedId = 0;

        private const float appearanceWeight = 0.775f;
        private const float smoothAppearanceWeight = 0.875f;

        public Matcher(string yoloPath, string fastReIDPath)
        {
            yoloPredictor = new YoloRecognizer(yoloPath);
            reidAppExtractor = new FastReID(fastReIDPath);

            tracks = [];

        }

        public void Dispose()
        {
            yoloPredictor.model.Dispose();
            reidAppExtractor.session.Dispose();
        }

        public (List<Track>, YoloDetection detections) Run(string path)
        {
            YoloDetection detections = yoloPredictor.recognize_image(path);

            List<Detail> detectionDetails = [];

            for (int i = 0; i < detections.Detections.Count; i++)
            {
                Detail details = reidAppExtractor.Recognize(detections.GetCut(i));

                detectionDetails.Add(details);
            }

            if(tracks.Count == 0)
            {
                Console.WriteLine("No tracks, adding new ones");
                AddNewTracks(detections.Detections, detectionDetails);
                return (tracks, detections);
            }

            Console.WriteLine($"Current tracks: {tracks.Count}");

            PredictBoundingBoxes();

            (List<(int, int)> matched, List<int> unmatched) = Match(detections, detectionDetails);

            Console.WriteLine($"Matched: {matched.Count}, Unmatched: {unmatched.Count}");

            UpdateTracks(matched, detections, detectionDetails);


            for(int i = 0; i<unmatched.Count; i++)
            {
                Rect bounds = YoloDetection.GetBounds(detections.Detections[unmatched[i]]);
                AddNewTrack(bounds, detectionDetails[unmatched[i]], trackedId++);
            }

            List<Track> confirmed = ConfirmTracks();

            Console.WriteLine($"Confirmed: {confirmed.Count}");

            RemoveOutdatedTracks();

            return (confirmed, detections);

        }

        private void AddNewTracks(List<ObjectDetection> detections, List<Detail> details )
        {
            for(int i = 0; i < detections.Count; i++)
            {
                AddNewTrack(YoloDetection.GetBounds(detections[i]), details[i], trackedId++);
            }
        }

        private void AddNewTrack(Rect bounds, Detail details, int id)
        {
            Track track = new(id, bounds, details, 2560);
            tracks.Add(track);
        }

        private void PredictBoundingBoxes()
        {
            List<Track> toRemove = [];

            foreach (Track track in tracks)
            {
                Rect predict = track.PredictNextBounds();

                if (predict.Right >= track.trackLimit)
                {
                    toRemove.Add(track);
                    Console.WriteLine("Removing outside of frame");
                }


            }

            if (toRemove.Count > 0)
            {
                tracks = [.. tracks.Except(toRemove)];
            }
        }

        private (List<(int, int)> matchedPairs,  List<int> unmatchedAppearances) Match(YoloDetection detections, List<Detail> details)
        {
            double[,] matrix = new double[tracks.Count, details.Count];


            for (int i = 0; i < tracks.Count; i++)
            {
                for (int j = 0; j < details.Count; j++)
                {
                    double metric = tracks[i].medianAppearance.CosineDistance(details[j]);

                    if(metric < double.Epsilon)
                    {
                        matrix[i, j] = 0;
                    }
                    else
                    {
                        matrix[i, j] = metric;
                    }

                    float weight = (tracks[i].lifeTime < 40 ? appearanceWeight : smoothAppearanceWeight);

                    matrix[i, j] *= weight;
                    matrix[i, j] += (1 - weight) * IntersectionOverUnionLoss(tracks[i].predictedNextBounds, YoloDetection.GetBounds(detections.Detections[j]));


                }
            }

            HungarianAlgorithm hungarian = new HungarianAlgorithm(matrix);

            int[] assignments = hungarian.Solve();

            List<int> allItemIndexes = [];
            List<int> matched = [];
            List<(int, int)> matchedPairs = [];
            List<int> unmatchedAppearances = [];
            List<int> unmatchedTracks = [];

            if(details.Count > tracks.Count)
            {
                for (int i = 0; i < details.Count; i++)
                    allItemIndexes.Add(i);

                for (int i = 0; i < tracks.Count; i++)
                    matched.Add(assignments[i]);

                unmatchedAppearances = [.. allItemIndexes.Except(matched)];
            }

            for (int i = 0; i < assignments.Length; i++)
            {
                if (assignments[i] == -1)
                {
                    unmatchedTracks.Add(i);
                }
                else if (1 - matrix[i, assignments[i]] < 0.5)
                {
                    unmatchedTracks.Add(i);
                    unmatchedAppearances.Add(assignments[i]);
                }
                else
                {
                    matchedPairs.Add((i, assignments[i]));
                }

            }

            return (matchedPairs, unmatchedAppearances);

        }

        private float IntersectionOverUnionLoss(Rect predict, Rect real) {
               
            Rect intersection = predict.Intersect(real);

            Rect union = predict.Union(real);

            float intersectArea = intersection.Width * intersection.Height;
            
            float unionArea = union.Width * union.Height;

            float inverseIntersecArea = unionArea - intersectArea;

            Console.WriteLine($"Prediction: ({predict.Left},{predict.Top})({predict.Right},{predict.Bottom})");
            Console.WriteLine($"Real: ({real.Left},{real.Top})({real.Right},{real.Bottom})");

            Console.WriteLine($"Intersect: {intersectArea}, Union: {unionArea}, Inverse: {inverseIntersecArea}");



            if (inverseIntersecArea < float.Epsilon)
            {
                return 0;
            }

            return 1 - (intersectArea / inverseIntersecArea);

        }

        private void UpdateTracks(List<(int trackIdx, int detIdx)> matched, YoloDetection detections, List<Detail> details)
        {
            for (int i = 0; i < matched.Count; i++)
            {
                int trackIdx = matched[i].trackIdx;
                int detailsIdx = matched[i].detIdx;

                Rect bounds = YoloDetection.GetBounds(detections.Detections[detailsIdx]);

                tracks[trackIdx].Register(bounds, details[detailsIdx]);

                tracks[trackIdx].Update();

            }
        }

        private List<Track> ConfirmTracks()
        {
            List<Track> confirmed = [];
            foreach (Track track in tracks)
            {
                if (track.consecutiveHits > 1 && track.missedFrames == 0)
                {
                    confirmed.Add(track);
                }
                else
                {
                    Console.WriteLine($"{track.id} Not confirmed: {track.consecutiveHits} hits, {track.missedFrames} missed");
                }
            }
            return confirmed;
        }

        private void RemoveOutdatedTracks()
        {
            List<Track> outdated = [];
            foreach (Track track in tracks)
            {
                if (track.missedFrames > 50)
                {
                    outdated.Add(track);
                }
            }

            if(outdated.Count > 0)
            {
                tracks = [.. tracks.Except(outdated)];
            }
        }
    }
}
