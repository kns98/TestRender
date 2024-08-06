using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace RayTracerGUI
{
    public struct Vector3f
    {
        public double X { get; }
        public double Y { get; }
        public double Z { get; }

        public Vector3f(double x, double y, double z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public double this[int index]
        {
            get
            {
                return index switch
                {
                    0 => X,
                    1 => Y,
                    2 => Z,
                    _ => throw new IndexOutOfRangeException(),
                };
            }
        }

        public static Vector3f Zero => new Vector3f(0, 0, 0);
        public static Vector3f OneX => new Vector3f(1, 0, 0);
        public static Vector3f OneY => new Vector3f(0, 1, 0);
        public static Vector3f OneZ => new Vector3f(0, 0, 1);

        public static Vector3f operator -(Vector3f a) => new Vector3f(-a.X, -a.Y, -a.Z);
        public static Vector3f operator -(Vector3f a, Vector3f b) => new Vector3f(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        public static Vector3f operator +(Vector3f a, Vector3f b) => new Vector3f(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        public static Vector3f operator *(Vector3f a, double scalar) => new Vector3f(a.X * scalar, a.Y * scalar, a.Z * scalar);
        public static Vector3f operator /(Vector3f a, double scalar) => new Vector3f(a.X / scalar, a.Y / scalar, a.Z / scalar);

        public static double Dot(Vector3f a, Vector3f b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        public static Vector3f Cross(Vector3f a, Vector3f b) => new Vector3f(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);

        public static Vector3f Unitize(Vector3f vector)
        {
            double length = Math.Sqrt(vector.X * vector.X + vector.Y * vector.Y + vector.Z * vector.Z);
            return vector / length;
        }

        public double Length() => Math.Sqrt(X * X + Y * Y + Z * Z);

        public static Vector3f Reflect(Vector3f incident, Vector3f normal)
        {
            return incident - normal * 2 * Dot(incident, normal);
        }

        public static Vector3f Refract(Vector3f incident, Vector3f normal, double eta)
        {
            double cosi = -Math.Max(-1.0, Math.Min(1.0, Dot(incident, normal)));
            double etai = 1, etat = eta;
            Vector3f n = normal;
            if (cosi < 0) { cosi = -cosi; etai = eta; etat = 1; n = -normal; }
            double etaRatio = etai / etat;
            double k = 1 - etaRatio * etaRatio * (1 - cosi * cosi);
            return k < 0 ? Zero : etaRatio * incident + (etaRatio * cosi - Math.Sqrt(k)) * n;
        }
    }

    public class Material
    {
        public Vector3f Color { get; set; }
        public double Reflectivity { get; set; }
        public double Emissivity { get; set; }
        public double Transparency { get; set; }
        public double RefractiveIndex { get; set; }
        public Texture Texture { get; set; }

        public Material(Vector3f color, double reflectivity, double emissivity, double transparency = 0.0, double refractiveIndex = 1.0, Texture texture = null)
        {
            Color = color;
            Reflectivity = reflectivity;
            Emissivity = emissivity;
            Transparency = transparency;
            RefractiveIndex = refractiveIndex;
            Texture = texture;
        }

        public Vector3f GetColor(double u, double v)
        {
            return Texture != null ? Texture.GetColor(u, v) : Color;
        }
    }

    public class Texture
    {
        private readonly Vector3f[,] pixels;
        public int Width { get; }
        public int Height { get; }

        public Texture(string filename)
        {
            using (StreamReader reader = new StreamReader(filename))
            {
                reader.ReadLine(); // P3
                string[] dimensions = reader.ReadLine().Split();
                Width = int.Parse(dimensions[0]);
                Height = int.Parse(dimensions[1]);
                reader.ReadLine(); // 255

                pixels = new Vector3f[Width, Height];
                for (int j = 0; j < Height; j++)
                {
                    for (int i = 0; i < Width; i++)
                    {
                        int r = int.Parse(reader.ReadLine());
                        int g = int.Parse(reader.ReadLine());
                        int b = int.Parse(reader.ReadLine());
                        pixels[i, j] = new Vector3f(r / 255.0, g / 255.0, b / 255.0);
                    }
                }
            }
        }

        public Vector3f GetColor(double u, double v)
        {
            int i = (int)((u * Width) % Width);
            int j = (int)(((1 - v) * Height) % Height);
            return pixels[i, j];
        }
    }

    public class Ray
    {
        public Vector3f Origin { get; }
        public Vector3f Direction { get; }

        public Ray(Vector3f origin, Vector3f direction)
        {
            Origin = origin;
            Direction = Vector3f.Unitize(direction);
        }

        public bool Intersects(Triangle triangle, out double distance)
        {
            const double EPSILON = 0.0000001;
            Vector3f vertex0 = triangle.Vertex0;
            Vector3f vertex1 = triangle.Vertex1;
            Vector3f vertex2 = triangle.Vertex2;

            Vector3f edge1 = vertex1 - vertex0;
            Vector3f edge2 = vertex2 - vertex0;

            Vector3f h = Vector3f.Cross(Direction, edge2);
            double a = Vector3f.Dot(edge1, h);
            if (a > -EPSILON && a < EPSILON)
            {
                distance = 0;
                return false;
            }

            double f = 1.0 / a;
            Vector3f s = Origin - vertex0;
            double u = f * Vector3f.Dot(s, h);
            if (u < 0.0 || u > 1.0)
            {
                distance = 0;
                return false;
            }

            Vector3f q = Vector3f.Cross(s, edge1);
            double v = f * Vector3f.Dot(Direction, q);
            if (v < 0.0 || u + v > 1.0)
            {
                distance = 0;
                return false;
            }

            double t = f * Vector3f.Dot(edge2, q);
            if (t > EPSILON)
            {
                distance = t;
                return true;
            }
            else
            {
                distance = 0;
                return false;
            }
        }
    }

    public class Triangle
    {
        public Vector3f Vertex0 { get; }
        public Vector3f Vertex1 { get; }
        public Vector3f Vertex2 { get; }
        private readonly Material material;
        private readonly Vector3f uv0;
        private readonly Vector3f uv1;
        private readonly Vector3f uv2;

        public Triangle(Vector3f vertex0, Vector3f vertex1, Vector3f vertex2, Material material, Vector3f uv0, Vector3f uv1, Vector3f uv2)
        {
            Vertex0 = vertex0;
            Vertex1 = vertex1;
            Vertex2 = vertex2;
            this.material = material;
            this.uv0 = uv0;
            this.uv1 = uv1;
            this.uv2 = uv2;
        }

        public Material GetMaterial()
        {
            return material;
        }

        public Vector3f Normal()
        {
            return Vector3f.Unitize(Vector3f.Cross(Vertex1 - Vertex0, Vertex2 - Vertex0));
        }

        public BoundingBox GetBoundingBox()
        {
            Vector3f small = new Vector3f(
                Math.Min(Vertex0.X, Math.Min(Vertex1.X, Vertex2.X)),
                Math.Min(Vertex0.Y, Math.Min(Vertex1.Y, Vertex2.Y)),
                Math.Min(Vertex0.Z, Math.Min(Vertex1.Z, Vertex2.Z))
            );

            Vector3f big = new Vector3f(
                Math.Max(Vertex0.X, Math.Max(Vertex1.X, Vertex2.X)),
                Math.Max(Vertex0.Y, Math.Max(Vertex1.Y, Vertex2.Y)),
                Math.Max(Vertex0.Z, Math.Max(Vertex1.Z, Vertex2.Z))
            );

            return new BoundingBox(small, big);
        }

        public bool Intersect(Ray ray, out double distance, out double u, out double v)
        {
            if (ray.Intersects(this, out distance))
            {
                Vector3f p = ray.Origin + ray.Direction * distance;
                Vector3f w = p - Vertex0;
                double denominator = Vector3f.Dot(Vector3f.Cross(Vertex1 - Vertex0, Vertex2 - Vertex0), Normal());
                u = Vector3f.Dot(Vector3f.Cross(Vertex2 - Vertex0, w), Normal()) / denominator;
                v = Vector3f.Dot(Vector3f.Cross(w, Vertex1 - Vertex0), Normal()) / denominator;
                return true;
            }
            u = v = 0;
            return false;
        }
    }

    public class Light
    {
        public Vector3f Position { get; set; }
        public Vector3f Color { get; set; }

        public Light(Vector3f position, Vector3f color)
        {
            Position = position;
            Color = color;
        }
    }

    public class Scene
    {
        private readonly List<Triangle> triangles;
        private readonly List<Light> lights;
        private BVHNode bvhRoot;

        public Scene()
        {
            triangles = new List<Triangle>();
            lights = new List<Light>();
        }

        public void AddTriangle(Triangle triangle)
        {
            triangles.Add(triangle);
            bvhRoot = new BVHNode(triangles, 0, triangles.Count);
        }

        public void AddLight(Light light)
        {
            lights.Add(light);
        }

        public IEnumerable<Triangle> GetTriangles() => triangles;
        public IEnumerable<Light> GetLights() => lights;

        public bool Intersect(Ray ray, out Triangle hitTriangle, out double hitDistance)
        {
            return bvhRoot.Intersect(ray, 0, double.MaxValue, out hitTriangle, out hitDistance);
        }
    }

    public class BVHNode
    {
        public BVHNode Left { get; private set; }
        public BVHNode Right { get; private set; }
        public BoundingBox Box { get; private set; }
        public Triangle Triangle { get; private set; }

        public BVHNode(List<Triangle> triangles, int start, int end)
        {
            var objects = new List<Triangle>(triangles);
            int axis = GetBestAxisToSplit(objects, start, end);
            objects.Sort((a, b) => a.GetBoundingBox().Min[axis].CompareTo(b.GetBoundingBox().Min[axis]));

            int objectSpan = end - start;

            if (objectSpan == 1)
            {
                Left = Right = new BVHNode(objects[start]);
            }
            else if (objectSpan == 2)
            {
                Left = new BVHNode(objects[start]);
                Right = new BVHNode(objects[start + 1]);
            }
            else
            {
                int mid = start + objectSpan / 2;
                Left = new BVHNode(objects, start, mid);
                Right = new BVHNode(objects, mid, end);
            }

            Box = BoundingBox.SurroundingBox(Left.Box, Right.Box);
        }

        private int GetBestAxisToSplit(List<Triangle> triangles, int start, int end)
        {
            int bestAxis = 0;
            double bestCost = double.MaxValue;
            for (int axis = 0; axis < 3; axis++)
            {
                triangles.Sort((a, b) => a.GetBoundingBox().Min[axis].CompareTo(b.GetBoundingBox().Min[axis]));
                BoundingBox boxLeft = triangles[start].GetBoundingBox();
                BoundingBox boxRight = triangles[end - 1].GetBoundingBox();
                double cost = 0;
                for (int i = start + 1; i < end; i++)
                {
                    boxLeft = BoundingBox.SurroundingBox(boxLeft, triangles[i].GetBoundingBox());
                    boxRight = BoundingBox.SurroundingBox(boxRight, triangles[i].GetBoundingBox());
                    double surfaceAreaLeft = boxLeft.GetSurfaceArea();
                    double surfaceAreaRight = boxRight.GetSurfaceArea();
                    cost = surfaceAreaLeft * (i - start) + surfaceAreaRight * (end - i);
                }
                if (cost < bestCost)
                {
                    bestAxis = axis;
                    bestCost = cost;
                }
            }
            return bestAxis;
        }

        private BVHNode(Triangle triangle)
        {
            Triangle = triangle;
            Box = triangle.GetBoundingBox();
        }

        public bool Intersect(Ray ray, double tMin, double tMax, out Triangle hitTriangle, out double hitDistance)
        {
            hitTriangle = null;
            hitDistance = double.MaxValue;
            if (!Box.Intersects(ray, tMin, tMax))
                return false;

            bool hitLeft = Left?.Intersect(ray, tMin, tMax, out Triangle leftTriangle, out double leftDistance) ?? false;
            bool hitRight = Right?.Intersect(ray, tMin, tMax, out Triangle rightTriangle, out double rightDistance) ?? false;

            if (hitLeft && hitRight)
            {
                if (leftDistance < rightDistance)
                {
                    hitTriangle = leftTriangle;
                    hitDistance = leftDistance;
                }
                else
                {
                    hitTriangle = rightTriangle;
                    hitDistance = rightDistance;
                }
                return true;
            }

            if (hitLeft)
            {
                hitTriangle = leftTriangle;
                hitDistance = leftDistance;
                return true;
            }

            if (hitRight)
            {
                hitTriangle = rightTriangle;
                hitDistance = rightDistance;
                return true;
            }

            return false;
        }
    }

    public class BoundingBox
    {
        public Vector3f Min { get; private set; }
        public Vector3f Max { get; private set; }

        public BoundingBox(Vector3f min, Vector3f max)
        {
            Min = min;
            Max = max;
        }

        public static BoundingBox SurroundingBox(BoundingBox box0, BoundingBox box1)
        {
            Vector3f small = new Vector3f(Math.Min(box0.Min.X, box1.Min.X),
                                          Math.Min(box0.Min.Y, box1.Min.Y),
                                          Math.Min(box0.Min.Z, box1.Min.Z));

            Vector3f big = new Vector3f(Math.Max(box0.Max.X, box1.Max.X),
                                        Math.Max(box0.Max.Y, box1.Max.Y),
                                        Math.Max(box0.Max.Z, box1.Max.Z));

            return new BoundingBox(small, big);
        }

        public bool Intersects(Ray ray, double tMin, double tMax)
        {
            for (int a = 0; a < 3; a++)
            {
                double invD = 1.0 / ray.Direction[a];
                double t0 = (Min[a] - ray.Origin[a]) * invD;
                double t1 = (Max[a] - ray.Origin[a]) * invD;

                if (invD < 0.0)
                {
                    double temp = t0;
                    t0 = t1;
                    t1 = temp;
                }

                tMin = t0 > tMin ? t0 : tMin;
                tMax = t1 < tMax ? t1 : tMax;

                if (tMax <= tMin)
                    return false;
            }
            return true;
        }

        public double GetSurfaceArea()
        {
            Vector3f diff = Max - Min;
            return 2 * (diff.X * diff.Y + diff.X * diff.Z + diff.Y * diff.Z);
        }
    }

    public class RayTracer
    {
        private readonly Scene scene;
        private readonly int maxDepth;
        private readonly int samplesPerPixel;

        public RayTracer(Scene scene, int maxDepth = 5, int samplesPerPixel = 100)
        {
            this.scene = scene;
            this.maxDepth = maxDepth;
            this.samplesPerPixel = samplesPerPixel;
        }

        public Vector3f Radiance(Vector3f position, Vector3f direction, Random random, int depth = 0)
        {
            if (depth > maxDepth)
                return Vector3f.Zero;

            Ray ray = new Ray(position, direction);
            if (scene.Intersect(ray, out Triangle hitTriangle, out double hitDistance))
            {
                Vector3f hitPoint = ray.Origin + ray.Direction * hitDistance;
                Material material = hitTriangle.GetMaterial();
                Vector3f normal = hitTriangle.Normal();
                double u, v;
                hitTriangle.Intersect(ray, out _, out u, out v);
                Vector3f color = material.Emissivity * material.GetColor(u, v);

                Vector3f directLighting = Vector3f.Zero;
                foreach (var light in scene.GetLights())
                {
                    Vector3f lightDirection = Vector3f.Unitize(light.Position - hitPoint);
                    Ray shadowRay = new Ray(hitPoint + normal * 0.001, lightDirection);

                    if (!scene.Intersect(shadowRay, out _, out _))
                    {
                        double lambertian = Math.Max(0, Vector3f.Dot(normal, lightDirection));
                        Vector3f reflectionDirection = Vector3f.Reflect(-lightDirection, normal);
                        double specular = Math.Pow(Math.Max(0, Vector3f.Dot(reflectionDirection, direction)), 32);

                        directLighting += light.Color * (material.GetColor(u, v) * lambertian + specular * material.Reflectivity);
                    }
                }

                Vector3f indirectLighting = Vector3f.Zero;
                for (int i = 0; i < samplesPerPixel; i++)
                {
                    Vector3f randomDirection = RandomHemisphereDirection(normal, random);
                    indirectLighting += Radiance(hitPoint + normal * 0.001, randomDirection, random, depth + 1);
                }
                indirectLighting /= samplesPerPixel;

                color += directLighting + indirectLighting * material.Color;
                return color;
            }

            return Vector3f.Zero;
        }

        private Vector3f RandomHemisphereDirection(Vector3f normal, Random random)
        {
            double u = random.NextDouble();
            double v = random.NextDouble();
            double theta = 2 * Math.PI * u;
            double phi = Math.Acos(2 * v - 1);
            double x = Math.Sin(phi) * Math.Cos(theta);
            double y = Math.Sin(phi) * Math.Sin(theta);
            double z = Math.Cos(phi);
            Vector3f randomDirection = new Vector3f(x, y, z);
            return Vector3f.Dot(randomDirection, normal) > 0 ? randomDirection : -randomDirection;
        }
    }

    public class RenderedImage
    {
        private readonly int width;
        private readonly int height;
        private readonly Vector3f[,] pixels;

        public RenderedImage(int width, int height)
        {
            this.width = width;
            this.height = height;
            pixels = new Vector3f[width, height];
        }

        public int Width => width;
        public int Height => height;
        public double AspectRatio => (double)width / height;

        public void AddToPixel(int x, int y, Vector3f color)
        {
            pixels[x, y] = color;
        }

        public Vector3f GetPixel(int x, int y)
        {
            return pixels[x, y];
        }

        public void SaveAsPPM(string filename)
        {
            using (StreamWriter writer = new StreamWriter(filename))
            {
                writer.WriteLine($"P3\n{width} {height}\n255");
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Vector3f color = pixels[x, y];
                        int r = (int)(Math.Min(1.0, color.X) * 255);
                        int g = (int)(Math.Min(1.0, color.Y) * 255);
                        int b = (int)(Math.Min(1.0, color.Z) * 255);
                        writer.WriteLine($"{r} {g} {b}");
                    }
                }
            }
        }
    }

    public static class ParallelHelper
    {
        public static void For2D(int startY, int endY, int startX, int endX, IAction2D action)
        {
            int batchHeight = (endY - startY) / Environment.ProcessorCount;

            Parallel.For(0, Environment.ProcessorCount, i =>
            {
                int currentStartY = startY + i * batchHeight;
                int currentEndY = (i == Environment.ProcessorCount - 1) ? endY : currentStartY + batchHeight;

                for (int y = currentStartY; y <= currentEndY; y++)
                {
                    for (int x = startX; x <= endX; x++)
                    {
                        action.Invoke(x, y);
                    }
                }
            });
        }
    }

    public interface IAction2D
    {
        void Invoke(int i, int j);
    }

    public class Camera
    {
        private readonly Vector3f viewPosition;
        private readonly double viewAngle;
        private readonly Vector3f viewDirection;
        private readonly Vector3f right;
        private readonly Vector3f up;

        public Camera(TextReader inBuffer)
        {
            viewPosition = Scanf.Read(inBuffer);
            viewDirection = Vector3f.Unitize(Scanf.Read(inBuffer));
            viewDirection = viewDirection.IsZero() ? Vector3f.OneZ : viewDirection;

            string line = Scanf.GetLine(inBuffer);
            double angle = Math.Max(10, Math.Min(160, double.Parse(line)));
            viewAngle = angle * (Math.PI / 180.0);

            right = Vector3f.Unitize(Vector3f.Cross(Vector3f.OneY, viewDirection));
            if (right.IsZero())
            {
                right = Vector3f.Unitize(Vector3f.Cross(viewDirection, Vector3f.OneX));
            }

            up = Vector3f.Unitize(Vector3f.Cross(viewDirection, right));
        }

        public Vector3f EyePoint => viewPosition;

        public RenderedImage Frame(Scene scene, RenderedImage renderedImage, Random random)
        {
            var rayTracer = new RayTracer(scene);
            var action = new Render2d(rayTracer, renderedImage, new Random(), up, right, viewDirection, viewAngle, viewPosition);

            ParallelHelper.For2D(0, renderedImage.Height - 1, 0, renderedImage.Width - 1, action);

            return renderedImage;
        }

        private class Render2d : IAction2D
        {
            private readonly RayTracer rayTracer;
            private readonly RenderedImage renderedImage;
            private readonly Random rand;
            private readonly Vector3f up;
            private readonly Vector3f right;
            private readonly Vector3f dir;
            private readonly double ang;
            private readonly Vector3f pos;
            private int count;
            private readonly int total;
            private readonly int samplesPerPixel;

            public Render2d(RayTracer r, RenderedImage i, Random ra, Vector3f up, Vector3f right, Vector3f dir, double ang, Vector3f pos, int samplesPerPixel = 4)
            {
                rayTracer = r;
                renderedImage = i;
                rand = ra;
                this.up = up;
                this.right = right;
                this.dir = dir;
                this.ang = ang;
                this.pos = pos;
                this.samplesPerPixel = samplesPerPixel;
                total = (renderedImage.Height - 1) * (renderedImage.Width - 1);
            }

            public void Invoke(int x, int y)
            {
                Vector3f color = Vector3f.Zero;

                for (int i = 0; i < samplesPerPixel; i++)
                {
                    double u = (x + rand.NextDouble()) / renderedImage.Width;
                    double v = (y + rand.NextDouble()) / renderedImage.Height;

                    double f1 = (u * 2 - 1) * Math.Tan(ang * 0.5) * renderedImage.AspectRatio;
                    double num1 = (v * 2 - 1) * Math.Tan(ang * 0.5);

                    var V_2 = right * f1 + up * num1;
                    var V_3 = Vector3f.Unitize(dir + V_2);
                    color += rayTracer.Radiance(pos, V_3, rand);
                }

                color /= samplesPerPixel;
                renderedImage.AddToPixel(x, y, color);

                int num2 = Interlocked.Increment(ref count);
                if (num2 % 100000 == 0)
                {
                    int V_7 = num2 / (total / 100);
                    Console.WriteLine($"{V_7} % of pixels processed: {DateTime.Now}");
                }
            }
        }
    }

    public static class Scanf
    {
        public static string GetLine(TextReader reader)
        {
            return reader.ReadLine();
        }

        public static Vector3f Read(TextReader reader)
        {
            string[] parts = reader.ReadLine().Split();
            return new Vector3f(double.Parse(parts[0]), double.Parse(parts[1]), double.Parse(parts[2]));
        }
    }

    public partial class MainForm : Form
    {
        private Scene scene;
        private Camera camera;
        private RenderedImage renderedImage;
        private RayTracer rayTracer;

        public MainForm()
        {
            InitializeComponent();
            InitializeScene();
        }

        private void InitializeScene()
        {
            int width = 800;
            int height = 600;
            renderedImage = new RenderedImage(width, height);
            scene = new Scene();
            scene.AddTriangle(new Triangle(new Vector3f(0, -1, 3), new Vector3f(1, 1, 3), new Vector3f(-1, 1, 3), new Material(new Vector3f(0.8, 0.3, 0.3), 0.5, 0.1), new Vector3f(0, 0), new Vector3f(1, 0), new Vector3f(0, 1)));
            scene.AddLight(new Light(new Vector3f(0, 5, 5), new Vector3f(1, 1, 1)));

            using (TextReader reader = new StringReader("0 0 0\n0 0 1\n90"))
            {
                camera = new Camera(reader);
            }

            rayTracer = new RayTracer(scene);
        }

        private async void RenderButton_Click(object sender, EventArgs e)
        {
            RenderButton.Enabled = false;
            await Task.Run(() => camera.Frame(scene, renderedImage, new Random()));
            RenderButton.Enabled = true;
            DisplayRenderedImage();
        }

        private void DisplayRenderedImage()
        {
            Bitmap bitmap = new Bitmap(renderedImage.Width, renderedImage.Height);
            for (int y = 0; y < renderedImage.Height; y++)
            {
                for (int x = 0; x < renderedImage.Width; x++)
                {
                    Vector3f color = renderedImage.GetPixel(x, y);
                    int r = (int)(Math.Min(1.0, color.X) * 255);
                    int g = (int)(Math.Min(1.0, color.Y) * 255);
                    int b = (int)(Math.Min(1.0, color.Z) * 255);
                    bitmap.SetPixel(x, y, Color.FromArgb(r, g, b));
                }
            }

            RenderedPictureBox.Image = bitmap;
        }
    }
}
