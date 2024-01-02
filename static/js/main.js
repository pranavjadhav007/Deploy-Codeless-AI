$(document).ready(function() {
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#imagePreview1').hide();
    $('#btn-predict').hide();


    function readURL(input, previewElement) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function(e) {
                previewElement.css('background-image', 'url(' + e.target.result + ')');
                previewElement.hide();
                previewElement.fadeIn(650);
                $('#btn-predict').show();
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload1").change(function() {
        $('.image-section').show();
        $('#btn-predict').hide();
        $('#result').hide();
        $('#result-text').hide();
        readURL(this, $('#imagePreview1'));
        $('#imagePreview1').show();
    });

    $('#btn-predict').click(function() {
        console.log("check console1")
        var form_data = new FormData($('#upload-file')[0]);

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
            success: function(data) {
                console.log("check console2")
                var resultText1 = data.result_text1;
                var resultText2 = data.result_text2;
                console.log(data);
                $('.loader').hide();
                $('#result').show();
                $('#result-text').html(resultText1 + '<br>' + resultText2);
                $('#result-text').show();
            }
        });
    });
});