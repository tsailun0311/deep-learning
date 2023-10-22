$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('.image-handle').hide();
    $("#drawingCanvas").hide();
    $("#saveButton").hide();
    $("#clearButton").hide();
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#canva").click(function () {
        $('.image-section').hide();
        $('.loader').hide();
        $('#result').hide();
        $('.image-handle').hide();
        $("#drawingCanvas").show();
        $("#saveButton").show();
        $("#clearButton").show();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'white'; // 设置填充颜色为白色
        ctx.fillRect(0, 0, canvas.width, canvas.height); // 以白色填充整个 Canvas
    });
    $("#upload-label").click(function () {
        $('.img-preview').css('width', '168');
        $('.img-preview').css('height', '168');
        $('.img-preview>div').css('background-size', '168px 168px');
        $('#h4').css('width', '168');
        $('.image-section').hide();
        $('.loader').hide();
        $('#result').hide();
        $('.image-handle').hide();
        $("#drawingCanvas").hide();
        $("#saveButton").hide();
        $("#clearButton").hide();
    });
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#result').text('');
        $('.image-handle').show();
        $('.image-handle').css("visibility", "hidden");
        $('#result').hide();
        readURL(this);
        var form_data = new FormData($('#upload-file')[0]);
        // Show loading animation
        $(this).hide();
        $('.loader').show();
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                var image = data.result[0];
                var data = data.result[1];
                // Get and display the result
                $('.loader').hide();
                var timestamp = new Date().getTime(); // 获取当前时间戳
                var updatedImagePath = image + '?t=' + timestamp; // 添加时间戳参数
                $('.imagehandle').css('background-image', 'url(' + updatedImagePath + ')');
                $('.image-handle').css("visibility", "visible");
                $('#result').fadeIn(600);
                $('#result').text(' 分析結果：' + data);
                console.log('Success!');
            },
        });
    });
    $("#saveButton").click(function () {
        $("#drawingCanvas").hide();
        $("#saveButton").hide();
        $("#clearButton").hide();
        $('.image-section').hide();
        $('#result').text('');
        $('.image-handle').hide();
        $('#result').hide();
        var canvas = document.getElementById("drawingCanvas");
        var imageData = canvas.toDataURL("image/png");
        $('.loader').show();
        // 将图像数据发送到服务器
        $.ajax({
            type: "POST",
            url: "/save_image",
            contentType: "application/json",
            data: JSON.stringify({ "image": imageData }),
            success: function (data) {
                var image = data.result[0];
                var data = data.result[1];
                // Get and display the result
                $('.loader').hide();
                var timestamp = new Date().getTime(); // 获取当前时间戳
                var updatedImagePath = image + '?t=' + timestamp; // 添加时间戳参数
                $('.img-preview').css('width', '336');
                $('.img-preview').css('height', '336');
                $('.img-preview>div').css('background-size', '336px 336px');
                $('#h4').css('width', '336');
                $('.imagehandle').css('background-image', 'url(' + updatedImagePath + ')');
                $('.image-handle').show();
                $('#result').fadeIn(600);
                $('#result').text(' 分析結果：' + data);
                console.log('Success!');
            },
        });
    });
});
const canvas = document.getElementById('drawingCanvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white'; // 设置填充颜色为白色
        ctx.fillRect(0, 0, canvas.width, canvas.height); // 以白色填充整个 Canvas
        let isDrawing = false;

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('touchstart', startDrawing);
        
        function startDrawing(e) {
            isDrawing = true;
            ctx.lineWidth = 10;  // 设置线条宽度
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
        
            const rect = canvas.getBoundingClientRect();  // 获取Canvas的边界矩形
            const offsetX = rect.left;
            const offsetY = rect.top;
        
            // 获取鼠标或触摸点在Canvas中的坐标
            const x = (e.clientX || e.touches[0].clientX) - offsetX;
            const y = (e.clientY || e.touches[0].clientY) - offsetY;
        
            ctx.beginPath();
            ctx.moveTo(x, y);
        
            // 更新绘图坐标
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('touchmove', draw);
        
            // 停止绘制
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('touchend', stopDrawing);
          }
        
        function draw(e) {
            if (!isDrawing) return;
        
            const rect = canvas.getBoundingClientRect();
            const offsetX = rect.left;
            const offsetY = rect.top;
            const x = (e.clientX || e.touches[0].clientX) - offsetX;
            const y = (e.clientY || e.touches[0].clientY) - offsetY;
        
            ctx.lineTo(x, y);
            ctx.stroke();
          }
        
        function stopDrawing() {
            isDrawing = false;
            canvas.removeEventListener('mousemove', draw);
            canvas.removeEventListener('touchmove', draw);
            canvas.removeEventListener('mouseup', stopDrawing);
            canvas.removeEventListener('touchend', stopDrawing);
        }

        // 清空画板
        document.getElementById('clearButton').addEventListener('click', () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = 'white'; // 设置填充颜色为白色
            ctx.fillRect(0, 0, canvas.width, canvas.height); // 以白色填充整个 Canvas
        });