using System.Drawing.Drawing2D;

namespace System.Windows.Forms
{
    public partial class ImageBox : PictureBox
    {
        protected override void OnPaint(PaintEventArgs pe)
        {
            var g = pe.Graphics;
            g.InterpolationMode = InterpolationMode.NearestNeighbor;
            g.SmoothingMode = SmoothingMode.None;
            g.PixelOffsetMode = PixelOffsetMode.Half;
            base.OnPaint(pe);
        }
    }
}
