
namespace ExploreCNN
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.numericUpDown_CONVOLUTION_NUMBER = new System.Windows.Forms.NumericUpDown();
            this.pictureBox_Image = new System.Windows.Forms.ImageBox();
            this.label_pooling2 = new System.Windows.Forms.Label();
            this.label_conv1 = new System.Windows.Forms.Label();
            this.label_pooling1 = new System.Windows.Forms.Label();
            this.label_conv2 = new System.Windows.Forms.Label();
            this.label_processimage = new System.Windows.Forms.Label();
            this.numericUpDown_image = new System.Windows.Forms.NumericUpDown();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.pictureBox_conv1 = new System.Windows.Forms.ImageBox();
            this.pictureBox_pooling1 = new System.Windows.Forms.ImageBox();
            this.pictureBox_conv2 = new System.Windows.Forms.ImageBox();
            this.pictureBox_pooling2 = new System.Windows.Forms.ImageBox();
            this.button_showcnn = new System.Windows.Forms.Button();
            this.button_train = new System.Windows.Forms.Button();
            this.button_loaddata = new System.Windows.Forms.Button();
            this.textBox_history = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown_CONVOLUTION_NUMBER)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_Image)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown_image)).BeginInit();
            this.tableLayoutPanel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_conv1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_pooling1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_conv2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_pooling2)).BeginInit();
            this.SuspendLayout();
            // 
            // numericUpDown_CONVOLUTION_NUMBER
            // 
            this.numericUpDown_CONVOLUTION_NUMBER.Font = new System.Drawing.Font("Microsoft YaHei UI", 21F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point);
            this.numericUpDown_CONVOLUTION_NUMBER.Location = new System.Drawing.Point(392, 69);
            this.numericUpDown_CONVOLUTION_NUMBER.Name = "numericUpDown_CONVOLUTION_NUMBER";
            this.numericUpDown_CONVOLUTION_NUMBER.Size = new System.Drawing.Size(134, 52);
            this.numericUpDown_CONVOLUTION_NUMBER.TabIndex = 17;
            this.numericUpDown_CONVOLUTION_NUMBER.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.numericUpDown_CONVOLUTION_NUMBER.Value = new decimal(new int[] {
            2,
            0,
            0,
            0});
            // 
            // pictureBox_Image
            // 
            this.pictureBox_Image.BackColor = System.Drawing.Color.DimGray;
            this.pictureBox_Image.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox_Image.Location = new System.Drawing.Point(532, 12);
            this.pictureBox_Image.Name = "pictureBox_Image";
            this.pictureBox_Image.Size = new System.Drawing.Size(123, 109);
            this.pictureBox_Image.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox_Image.TabIndex = 16;
            this.pictureBox_Image.TabStop = false;
            // 
            // label_pooling2
            // 
            this.label_pooling2.AutoSize = true;
            this.label_pooling2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.label_pooling2.Location = new System.Drawing.Point(509, 1);
            this.label_pooling2.Name = "label_pooling2";
            this.label_pooling2.Size = new System.Drawing.Size(134, 35);
            this.label_pooling2.TabIndex = 20;
            this.label_pooling2.Text = "Pooling2";
            this.label_pooling2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label_conv1
            // 
            this.label_conv1.AutoSize = true;
            this.label_conv1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.label_conv1.Location = new System.Drawing.Point(95, 1);
            this.label_conv1.Name = "label_conv1";
            this.label_conv1.Size = new System.Drawing.Size(131, 35);
            this.label_conv1.TabIndex = 16;
            this.label_conv1.Text = "Conv1";
            this.label_conv1.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label_pooling1
            // 
            this.label_pooling1.AutoSize = true;
            this.label_pooling1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.label_pooling1.Location = new System.Drawing.Point(233, 1);
            this.label_pooling1.Name = "label_pooling1";
            this.label_pooling1.Size = new System.Drawing.Size(131, 35);
            this.label_pooling1.TabIndex = 17;
            this.label_pooling1.Text = "Pooling1";
            this.label_pooling1.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label_conv2
            // 
            this.label_conv2.AutoSize = true;
            this.label_conv2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.label_conv2.Location = new System.Drawing.Point(371, 1);
            this.label_conv2.Name = "label_conv2";
            this.label_conv2.Size = new System.Drawing.Size(131, 35);
            this.label_conv2.TabIndex = 18;
            this.label_conv2.Text = "Conv2";
            this.label_conv2.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // label_processimage
            // 
            this.label_processimage.AutoSize = true;
            this.label_processimage.Dock = System.Windows.Forms.DockStyle.Fill;
            this.label_processimage.Location = new System.Drawing.Point(4, 37);
            this.label_processimage.Name = "label_processimage";
            this.label_processimage.Size = new System.Drawing.Size(84, 133);
            this.label_processimage.TabIndex = 19;
            this.label_processimage.Text = "Process Image";
            this.label_processimage.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // numericUpDown_image
            // 
            this.numericUpDown_image.Font = new System.Drawing.Font("Microsoft YaHei UI", 21F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point);
            this.numericUpDown_image.Location = new System.Drawing.Point(392, 12);
            this.numericUpDown_image.Maximum = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            this.numericUpDown_image.Name = "numericUpDown_image";
            this.numericUpDown_image.Size = new System.Drawing.Size(134, 52);
            this.numericUpDown_image.TabIndex = 18;
            this.numericUpDown_image.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.numericUpDown_image.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.CellBorderStyle = System.Windows.Forms.TableLayoutPanelCellBorderStyle.Single;
            this.tableLayoutPanel1.ColumnCount = 5;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 90F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 25F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 25F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 25F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 25F));
            this.tableLayoutPanel1.Controls.Add(this.label_pooling2, 4, 0);
            this.tableLayoutPanel1.Controls.Add(this.pictureBox_conv1, 1, 1);
            this.tableLayoutPanel1.Controls.Add(this.pictureBox_pooling1, 2, 1);
            this.tableLayoutPanel1.Controls.Add(this.pictureBox_conv2, 3, 1);
            this.tableLayoutPanel1.Controls.Add(this.pictureBox_pooling2, 4, 1);
            this.tableLayoutPanel1.Controls.Add(this.label_conv1, 1, 0);
            this.tableLayoutPanel1.Controls.Add(this.label_pooling1, 2, 0);
            this.tableLayoutPanel1.Controls.Add(this.label_conv2, 3, 0);
            this.tableLayoutPanel1.Controls.Add(this.label_processimage, 0, 1);
            this.tableLayoutPanel1.Location = new System.Drawing.Point(12, 553);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 35F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(647, 171);
            this.tableLayoutPanel1.TabIndex = 15;
            // 
            // pictureBox_conv1
            // 
            this.pictureBox_conv1.BackColor = System.Drawing.Color.DimGray;
            this.pictureBox_conv1.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox_conv1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBox_conv1.Location = new System.Drawing.Point(95, 40);
            this.pictureBox_conv1.Name = "pictureBox_conv1";
            this.pictureBox_conv1.Size = new System.Drawing.Size(131, 127);
            this.pictureBox_conv1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox_conv1.TabIndex = 4;
            this.pictureBox_conv1.TabStop = false;
            // 
            // pictureBox_pooling1
            // 
            this.pictureBox_pooling1.BackColor = System.Drawing.Color.DimGray;
            this.pictureBox_pooling1.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox_pooling1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBox_pooling1.Location = new System.Drawing.Point(233, 40);
            this.pictureBox_pooling1.Name = "pictureBox_pooling1";
            this.pictureBox_pooling1.Size = new System.Drawing.Size(131, 127);
            this.pictureBox_pooling1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox_pooling1.TabIndex = 5;
            this.pictureBox_pooling1.TabStop = false;
            // 
            // pictureBox_conv2
            // 
            this.pictureBox_conv2.BackColor = System.Drawing.Color.DimGray;
            this.pictureBox_conv2.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox_conv2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBox_conv2.Location = new System.Drawing.Point(371, 40);
            this.pictureBox_conv2.Name = "pictureBox_conv2";
            this.pictureBox_conv2.Size = new System.Drawing.Size(131, 127);
            this.pictureBox_conv2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox_conv2.TabIndex = 6;
            this.pictureBox_conv2.TabStop = false;
            // 
            // pictureBox_pooling2
            // 
            this.pictureBox_pooling2.BackColor = System.Drawing.Color.DimGray;
            this.pictureBox_pooling2.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox_pooling2.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBox_pooling2.Location = new System.Drawing.Point(509, 40);
            this.pictureBox_pooling2.Name = "pictureBox_pooling2";
            this.pictureBox_pooling2.Size = new System.Drawing.Size(134, 127);
            this.pictureBox_pooling2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox_pooling2.TabIndex = 7;
            this.pictureBox_pooling2.TabStop = false;
            // 
            // button_showcnn
            // 
            this.button_showcnn.Font = new System.Drawing.Font("Microsoft YaHei UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point);
            this.button_showcnn.Location = new System.Drawing.Point(260, 12);
            this.button_showcnn.Name = "button_showcnn";
            this.button_showcnn.Size = new System.Drawing.Size(126, 109);
            this.button_showcnn.TabIndex = 13;
            this.button_showcnn.Text = "Test CNN";
            this.button_showcnn.UseVisualStyleBackColor = true;
            this.button_showcnn.Click += new System.EventHandler(this.button_showcnn_Click);
            // 
            // button_train
            // 
            this.button_train.Font = new System.Drawing.Font("Microsoft YaHei UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point);
            this.button_train.Location = new System.Drawing.Point(136, 12);
            this.button_train.Name = "button_train";
            this.button_train.Size = new System.Drawing.Size(118, 109);
            this.button_train.TabIndex = 12;
            this.button_train.Text = "Train";
            this.button_train.UseVisualStyleBackColor = true;
            this.button_train.Click += new System.EventHandler(this.button_train_Click);
            // 
            // button_loaddata
            // 
            this.button_loaddata.Font = new System.Drawing.Font("Microsoft YaHei UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point);
            this.button_loaddata.Location = new System.Drawing.Point(12, 12);
            this.button_loaddata.Name = "button_loaddata";
            this.button_loaddata.Size = new System.Drawing.Size(118, 109);
            this.button_loaddata.TabIndex = 11;
            this.button_loaddata.Text = "Load Data";
            this.button_loaddata.UseVisualStyleBackColor = true;
            this.button_loaddata.Click += new System.EventHandler(this.button_loaddata_Click);
            // 
            // textBox_history
            // 
            this.textBox_history.Location = new System.Drawing.Point(12, 127);
            this.textBox_history.Multiline = true;
            this.textBox_history.Name = "textBox_history";
            this.textBox_history.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.textBox_history.Size = new System.Drawing.Size(647, 420);
            this.textBox_history.TabIndex = 14;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(670, 736);
            this.Controls.Add(this.numericUpDown_CONVOLUTION_NUMBER);
            this.Controls.Add(this.pictureBox_Image);
            this.Controls.Add(this.numericUpDown_image);
            this.Controls.Add(this.tableLayoutPanel1);
            this.Controls.Add(this.button_showcnn);
            this.Controls.Add(this.button_train);
            this.Controls.Add(this.button_loaddata);
            this.Controls.Add(this.textBox_history);
            this.Name = "Form1";
            this.Text = "MNIST CNN";
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown_CONVOLUTION_NUMBER)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_Image)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numericUpDown_image)).EndInit();
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_conv1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_pooling1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_conv2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox_pooling2)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.NumericUpDown numericUpDown_CONVOLUTION_NUMBER;
        private System.Windows.Forms.ImageBox pictureBox_Image;
        private System.Windows.Forms.Label label_pooling2;
        private System.Windows.Forms.Label label_conv1;
        private System.Windows.Forms.Label label_pooling1;
        private System.Windows.Forms.Label label_conv2;
        private System.Windows.Forms.Label label_processimage;
        private System.Windows.Forms.NumericUpDown numericUpDown_image;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.ImageBox pictureBox_conv1;
        private System.Windows.Forms.ImageBox pictureBox_pooling1;
        private System.Windows.Forms.ImageBox pictureBox_conv2;
        private System.Windows.Forms.ImageBox pictureBox_pooling2;
        private System.Windows.Forms.Button button_showcnn;
        private System.Windows.Forms.Button button_train;
        private System.Windows.Forms.Button button_loaddata;
        private System.Windows.Forms.TextBox textBox_history;
    }
}

