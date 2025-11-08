<?php
session_start();
$conn = new mysqli("localhost", "root", "", "website_login");

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if (!isset($_SESSION['users'])) {
    header("Location: signin.html");
    exit();
}

$username = $_SESSION['users'];
$sql = "SELECT * FROM users WHERE username='$username'";
$result = $conn->query($sql);
$user = $result->fetch_assoc();

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $full_name = $conn->real_escape_string($_POST['full_name']);
    $age = $conn->real_escape_string($_POST['age']);
    $gender = $conn->real_escape_string($_POST['gender']);
    $bio = $conn->real_escape_string($_POST['bio']);
    
    // Handle Profile Picture Upload
    if (!empty($_FILES["profile_picture"]["name"])) {
        $target_dir = "uploads/";
        $target_file = $target_dir . basename($_FILES["profile_picture"]["name"]);
        move_uploaded_file($_FILES["profile_picture"]["tmp_name"], $target_file);
        $update_picture = ", profile_picture='$target_file'";
    } else {
        $update_picture = "";
    }

    // Update user details
    $sql = "UPDATE users SET full_name='$full_name', age='$age', gender='$gender', bio='$bio' $update_picture WHERE username='$username'";
    
    if ($conn->query($sql) === TRUE) {
        echo "<script>alert('Profile updated!'); window.location.href='profile.html';</script>";
    } else {
        echo "Error updating profile: " . $conn->error;
    }
}

$conn->close();
?>
